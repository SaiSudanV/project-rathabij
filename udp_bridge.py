"""
udp_bridge.py — Godot ↔ PyTorch Real-Time Communication
========================================================
Zero-latency UDP socket bridge for real-time game state
streaming between the PyTorch AI brain and the Godot engine.

Protocol:
  - Godot sends JSON state packets to PyTorch every frame.
  - PyTorch responds with JSON action packets.
  - On connection, Godot sends an "action_map" dictionary
    so the debug logger can translate slot numbers to move names.

Packet Format (Godot → Python):
{
    "type": "state",
    "timestamp": 1234567890.123,
    "game_state": {
        "enemy_pos": [x, y],
        "player_pos": [x, y],
        "enemy_hp": 100,
        "player_hp": 75,
        "player_velocity": [vx, vy],
        "enemy_velocity": [vx, vy],
        "distance": 5.2,
        "player_is_attacking": false,
        "player_is_reloading": true,
        "projectiles": [...],
        "cooldowns": [0.0, 1.2, ...],
        "map_features": [...]
    },
    "events": ["player_missed", "enemy_took_damage"],
    "player_speech_text": "nice try loser"
}

Packet Format (Python → Godot):
{
    "type": "action",
    "timestamp": 1234567890.456,
    "actions": [0, 1, 0, 0, 1, ...],   // 40-slot binary array
    "chat_message": "you're predictable",
    "emotion": "confident",
    "debug": {
        "top_action": "dodge_left + attack_3",
        "confidence": 0.89,
        "mood": "confident"
    }
}

Handshake Packet (Godot → Python, sent once on connect):
{
    "type": "handshake",
    "action_map": {
        "0": "move_left",
        "1": "move_right",
        ...
        "39": "unused"
    },
    "state_dim": 64,
    "character_name": "Shadow Knight"
}
"""

from __future__ import annotations
import asyncio
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml


@dataclass
class GamePacket:
    """Parsed game state packet from Godot."""
    timestamp: float
    game_state: dict
    events: list[str] = field(default_factory=list)
    player_speech_text: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class ActionPacket:
    """Action response packet to send to Godot."""
    actions: list[int]            # 40-slot binary array
    chat_message: str = ""
    emotion: str = "neutral"
    debug: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "type": "action",
            "timestamp": time.time(),
            "actions": self.actions,
            "chat_message": self.chat_message,
            "emotion": self.emotion,
            "debug": self.debug,
        })


class UDPBridge:
    """
    Bidirectional UDP bridge between PyTorch and Godot.

    This class runs a UDP server that:
      1. Listens for state packets from Godot.
      2. Passes them to a callback (the AI brain).
      3. Sends back action packets to Godot.

    Usage:
        bridge = UDPBridge("config.yaml")

        def on_state(packet: GamePacket) -> ActionPacket:
            # Process with AI and return actions
            return ActionPacket(actions=[0, 1, 0, ...], chat_message="ez")

        bridge.start(on_state_callback=on_state)
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        net_cfg = cfg.get("network", {})
        self.host = net_cfg.get("host", "127.0.0.1")
        self.port = net_cfg.get("port", 9877)
        self.godot_port = net_cfg.get("godot_port", 9878)

        self.sock: socket.socket | None = None
        self.running = False
        self._thread: threading.Thread | None = None

        # Action map from Godot handshake (slot_id → move_name)
        self.action_map: dict[str, str] = {}
        self.character_name: str = "Unknown"
        self.state_dim: int = 64
        self._handshake_received = False

        # Stats
        self._packets_received = 0
        self._packets_sent = 0
        self._last_latency_ms = 0.0
        self._godot_addr: tuple[str, int] | None = None

    def start(
        self,
        on_state_callback: Callable[[GamePacket], ActionPacket],
        blocking: bool = False,
    ) -> None:
        """
        Start the UDP bridge.

        Args:
            on_state_callback: Function called with each GamePacket,
                               must return an ActionPacket.
            blocking: If True, blocks the calling thread.
        """
        self.running = True
        self._callback = on_state_callback

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.1)  # 100ms timeout for clean shutdown

        print(f"[UDP Bridge] Listening on {self.host}:{self.port}")
        print(f"[UDP Bridge] Will respond to Godot on port {self.godot_port}")

        if blocking:
            self._listen_loop()
        else:
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()

    def _listen_loop(self) -> None:
        """Main receive loop."""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65536)  # Max UDP datagram
                self._godot_addr = addr
                self._handle_packet(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[UDP Bridge] Error: {e}")

    def _handle_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse and dispatch a received packet."""
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return

        packet_type = payload.get("type", "")

        if packet_type == "handshake":
            self._handle_handshake(payload)
            return

        if packet_type == "state":
            recv_time = time.time()
            self._packets_received += 1

            # Parse into GamePacket
            packet = GamePacket(
                timestamp=payload.get("timestamp", recv_time),
                game_state=payload.get("game_state", {}),
                events=payload.get("events", []),
                player_speech_text=payload.get("player_speech_text", ""),
                raw=payload,
            )

            # Call the AI brain
            response = self._callback(packet)

            # Send response back to Godot
            self._send_response(response, addr)

            # Track latency
            self._last_latency_ms = (time.time() - recv_time) * 1000

    def _handle_handshake(self, payload: dict) -> None:
        """Process the initial handshake from Godot."""
        self.action_map = payload.get("action_map", {})
        self.character_name = payload.get("character_name", "Unknown")
        self.state_dim = payload.get("state_dim", 64)
        self._handshake_received = True

        print(f"[UDP Bridge] Handshake received!")
        print(f"  Character: {self.character_name}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Actions mapped: {len(self.action_map)}")

        # Send acknowledgment
        ack = json.dumps({
            "type": "handshake_ack",
            "status": "ready",
            "num_action_slots": 40,
        }).encode("utf-8")
        if self._godot_addr:
            self.sock.sendto(ack, self._godot_addr)

    def _send_response(self, response: ActionPacket, addr: tuple[str, int]) -> None:
        """Send an action packet back to Godot."""
        data = response.to_json().encode("utf-8")
        self.sock.sendto(data, (addr[0], self.godot_port))
        self._packets_sent += 1

    def translate_action(self, slot_index: int) -> str:
        """
        Translate a slot index to the character's actual move name.
        Used by the debug logger.
        """
        return self.action_map.get(str(slot_index), f"slot_{slot_index}")

    def translate_actions(self, action_binary: list[int]) -> list[str]:
        """Translate a full action vector to a list of active move names."""
        active = []
        for i, pressed in enumerate(action_binary):
            if pressed:
                active.append(self.translate_action(i))
        return active

    def stop(self) -> None:
        """Stop the UDP bridge."""
        self.running = False
        if self.sock:
            self.sock.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[UDP Bridge] Stopped.")

    def get_stats(self) -> dict:
        """Get bridge statistics for debugging."""
        return {
            "running": self.running,
            "handshake_received": self._handshake_received,
            "character": self.character_name,
            "packets_received": self._packets_received,
            "packets_sent": self._packets_sent,
            "last_latency_ms": f"{self._last_latency_ms:.2f}",
            "action_map_size": len(self.action_map),
        }
