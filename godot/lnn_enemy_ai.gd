extends Node

## ============================================================
## LNN Enemy AI — Godot UDP Client
## ============================================================
## Attach this script to any CharacterBody2D/3D enemy node.
## It streams game state to the Python LNN brain and applies
## the returned actions to the character.
##
## Setup:
##   1. Attach this script to your enemy character node.
##   2. Configure the action_map dictionary to match your
##      character's moves.
##   3. Set the player_node path to your player character.
##   4. Run the Python brain server first, then start the game.
## ============================================================

class_name LNNEnemyAI

## ── Configuration ──────────────────────────────────────────

## Path to the player character node
@export var player_node_path: NodePath = ""

## Network settings (must match config.yaml)
@export var python_host: String = "127.0.0.1"
@export var python_port: int = 9877
@export var listen_port: int = 9878

## How often to send state updates (times per second)
@export var update_rate: float = 30.0

## Character name (shown in debug logger)
@export var character_name: String = "Enemy Fighter"

## ── Internal State ─────────────────────────────────────────

var _udp := PacketPeerUDP.new()
var _listener := PacketPeerUDP.new()
var _player: Node = null
var _connected: bool = false
var _update_timer: float = 0.0
var _update_interval: float = 1.0 / 30.0

## Current actions from the LNN (40-slot binary array)
var current_actions: Array[int] = []

## Current actions from the Player (40-slot binary array)
## Set this array from your player script to give the AI perfect granular awareness!
var player_actions: Array[int] = []

## Latest chat message from the AI
var latest_chat_message: String = ""

## Latest mood
var latest_mood: String = "neutral"

## Match events to send next frame
var _pending_events: Array[String] = []

## Player speech text to send next frame
var _pending_speech: String = ""

## ── Action Map ─────────────────────────────────────────────
## Map action slot indices to your character's actual moves.
## Customize this for each character! Only map the slots you need.
## Unmapped slots are automatically ignored by the LNN.

var action_map: Dictionary = {
	"0": "move_up",
	"1": "move_down",
	"2": "move_left",
	"3": "move_right",
	"4": "light_attack_1",
	"5": "light_attack_2",
	"6": "light_attack_3",
	"7": "light_attack_4",
	"8": "light_attack_5",
	"9": "light_attack_6",
	"10": "heavy_attack_1",
	"11": "heavy_attack_2",
	"12": "heavy_attack_3",
	"13": "heavy_attack_4",
	"14": "heavy_attack_5",
	"15": "heavy_attack_6",
	"16": "special_1",
	"17": "special_2",
	"18": "special_3",
	"19": "special_4",
	"20": "special_5",
	"21": "special_6",
	"22": "block",
	"23": "dodge",
	"24": "parry",
	"25": "defensive_4",
	"26": "defensive_5",
	"27": "defensive_6",
	"28": "dash",
	"29": "jump",
	"30": "roll",
	"31": "movement_4",
	"32": "movement_5",
	"33": "movement_6",
	"34": "reserved_1",
	"35": "reserved_2",
	"36": "reserved_3",
	"37": "reserved_4",
	"38": "reserved_5",
	"39": "reserved_6",
}


## ── Lifecycle ──────────────────────────────────────────────

func _ready() -> void:
	# Initialize action arrays
	current_actions.resize(40)
	current_actions.fill(0)
	
	player_actions.resize(40)
	player_actions.fill(0)
	
	_update_interval = 1.0 / update_rate
	
	# Get player reference
	if player_node_path:
		_player = get_node_or_null(player_node_path)
	
	if _player == null:
		push_warning("LNNEnemyAI: No player node set! Set player_node_path.")
	
	# Setup UDP sender (to Python)
	_udp.connect_to_host(python_host, python_port)
	
	# Setup UDP listener (from Python)
	_listener.bind(listen_port)
	
	_connected = true
	print("[LNN] Connected to Python brain at %s:%d" % [python_host, python_port])
	
	# Send handshake
	_send_handshake()


func _process(delta: float) -> void:
	if not _connected:
		return
	
	# Check for responses from Python
	_receive_actions()
	
	# Send state updates at configured rate
	_update_timer += delta
	if _update_timer >= _update_interval:
		_update_timer = 0.0
		_send_state()
	
	# Apply the current actions to the character
	_apply_actions(delta)


func _exit_tree() -> void:
	_udp.close()
	_listener.close()


## ── Network: Sending ───────────────────────────────────────

func _send_handshake() -> void:
	"""Send the initial handshake with action map to Python."""
	var packet := {
		"type": "handshake",
		"action_map": action_map,
		"state_dim": 64,
		"character_name": character_name,
	}
	var json_str := JSON.stringify(packet)
	_udp.put_packet(json_str.to_utf8_buffer())
	print("[LNN] Handshake sent: %s (%d actions mapped)" % [character_name, action_map.size()])


func _send_state() -> void:
	"""Send the current game state to the Python brain."""
	if _player == null:
		return
	
	var enemy_pos := (owner as Node2D).global_position if owner is Node2D else Vector2.ZERO
	var player_pos := (_player as Node2D).global_position if _player is Node2D else Vector2.ZERO
	
	# Build state packet
	var game_state := {
		"enemy_pos": [enemy_pos.x, enemy_pos.y],
		"player_pos": [player_pos.x, player_pos.y],
		"enemy_hp": _get_hp(owner),
		"player_hp": _get_hp(_player),
		"player_velocity": _get_velocity(_player),
		"enemy_velocity": _get_velocity(owner),
		"distance": enemy_pos.distance_to(player_pos),
		"player_actions": player_actions,
	}
	
	var packet := {
		"type": "state",
		"timestamp": Time.get_unix_time_from_system(),
		"game_state": game_state,
		"events": _pending_events.duplicate(),
		"player_speech_text": _pending_speech,
	}
	
	var json_str := JSON.stringify(packet)
	_udp.put_packet(json_str.to_utf8_buffer())
	
	# Clear pending events
	_pending_events.clear()
	_pending_speech = ""


## ── Network: Receiving ─────────────────────────────────────

func _receive_actions() -> void:
	"""Check for and process action packets from Python."""
	while _listener.get_available_packet_count() > 0:
		var data := _listener.get_packet()
		var json_str := data.get_string_from_utf8()
		
		var json := JSON.new()
		var err := json.parse(json_str)
		if err != OK:
			continue
		
		var packet: Dictionary = json.data
		
		if packet.get("type") == "action":
			# Update action array
			var actions: Array = packet.get("actions", [])
			for i in range(min(actions.size(), 40)):
				current_actions[i] = int(actions[i])
			
			# Update chat message
			var chat := packet.get("chat_message", "")
			if chat != "":
				latest_chat_message = chat
				_on_chat_received(chat)
			
			# Update mood
			latest_mood = packet.get("emotion", "neutral")
		
		elif packet.get("type") == "handshake_ack":
			print("[LNN] Brain acknowledged handshake: %s" % packet.get("status", "?"))


## ── Action Application ─────────────────────────────────────
## Override these methods to connect LNN actions to your
## character's actual abilities.

func _apply_actions(delta: float) -> void:
	"""
	Apply the current 40-slot action array to the character.
	
	Override this method to implement your character's specific
	move mappings. Below is a template.
	"""
	var body := owner as CharacterBody2D
	if body == null:
		return
	
	var velocity := Vector2.ZERO
	var speed := 200.0  # Adjust to your character's speed
	
	# Movement (slots 0-3)
	if current_actions[0]:  # move_up
		velocity.y -= speed
	if current_actions[1]:  # move_down
		velocity.y += speed
	if current_actions[2]:  # move_left
		velocity.x -= speed
	if current_actions[3]:  # move_right
		velocity.x += speed
	
	body.velocity = velocity
	body.move_and_slide()
	
	# Light attacks (slots 4-9)
	for i in range(4, 10):
		if current_actions[i]:
			_execute_attack("light", i - 4)
	
	# Heavy attacks (slots 10-15)
	for i in range(10, 16):
		if current_actions[i]:
			_execute_attack("heavy", i - 10)
	
	# Special attacks (slots 16-21)
	for i in range(16, 22):
		if current_actions[i]:
			_execute_attack("special", i - 16)
	
	# Defensive (slots 22-27)
	if current_actions[22]:
		_execute_defense("block")
	if current_actions[23]:
		_execute_defense("dodge")
	if current_actions[24]:
		_execute_defense("parry")
	
	# Movement abilities (slots 28-33)
	if current_actions[28]:
		_execute_movement("dash")
	if current_actions[29]:
		_execute_movement("jump")
	if current_actions[30]:
		_execute_movement("roll")


## ── Override These Methods ──────────────────────────────────
## Implement these to connect to your character's animation
## system, hitboxes, and abilities.

func _execute_attack(type: String, variant: int) -> void:
	"""Called when the LNN wants to attack. Override this."""
	pass  # Example: owner.get_node("AnimPlayer").play("attack_%s_%d" % [type, variant])


func _execute_defense(type: String) -> void:
	"""Called when the LNN wants to defend. Override this."""
	pass  # Example: owner.get_node("Shield").activate()


func _execute_movement(type: String) -> void:
	"""Called when the LNN wants a movement ability. Override this."""
	pass  # Example: owner.dash(velocity.normalized() * dash_speed)


func _on_chat_received(message: String) -> void:
	"""Called when the AI generates trash talk. Override to display it."""
	print("[LNN Chat] %s" % message)
	# Example: owner.get_node("ChatBubble").show_message(message)


## ── Helper Methods ─────────────────────────────────────────

func _get_hp(node: Node) -> float:
	"""Try to get HP from a node. Override if your HP system is different."""
	if node.has_method("get_hp"):
		return node.get_hp()
	if "hp" in node:
		return node.hp
	if "health" in node:
		return node.health
	return 100.0


func _get_velocity(node: Node) -> Array:
	"""Get velocity from a node."""
	if node is CharacterBody2D:
		var v: Vector2 = (node as CharacterBody2D).velocity
		return [v.x, v.y]
	return [0.0, 0.0]


## ── Public API ─────────────────────────────────────────────

func report_event(event_name: String) -> void:
	"""
	Report a game event to the AI brain.
	Call this from your game logic when something notable happens.
	
	Examples:
		lnn_ai.report_event("got_kill")
		lnn_ai.report_event("player_missed")
		lnn_ai.report_event("took_damage")
		lnn_ai.report_event("combo_landed")
	"""
	_pending_events.append(event_name)


func report_player_speech(text: String) -> void:
	"""
	Forward transcribed player speech to the AI.
	Call this from your Moonshine STT callback.
	"""
	_pending_speech = text


func get_active_actions() -> Array[String]:
	"""Get a list of currently active action names (for debugging)."""
	var active: Array[String] = []
	for i in range(40):
		if current_actions[i]:
			var name: String = action_map.get(str(i), "slot_%d" % i)
			active.append(name)
	return active
