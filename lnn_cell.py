"""
lnn_cell.py — Liquid Neural Network Cell Wrapper
=================================================
Wraps the ncps (Neural Circuit Policies) CfC cell into a
clean PyTorch module that handles:
  - Continuous-time stepping with variable dt
  - Hidden state management
  - Batched sequence processing
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class LiquidCell(nn.Module):
    """
    A wrapper around the ncps CfC (Closed-form Continuous-time) cell.

    This is the core "brain" that processes inputs through a liquid
    neural circuit with continuous-time dynamics. The hidden state
    evolves smoothly over time, giving the network temporal memory
    without explicit LSTM-style gating.

    Args:
        input_size:  Dimension of the input feature vector.
        hidden_size: Dimension of the internal liquid state.
        num_units:   Number of inter-neurons in the NCP wiring.
    """

    def __init__(self, input_size: int, hidden_size: int, num_units: int = 64):
        super().__init__()
        self.hidden_size = hidden_size

        # AutoNCP requires output_size < units - 2.
        # We use more internal units than the output to satisfy
        # this constraint, then project back to hidden_size.
        ncp_units = hidden_size + 16  # Extra inter-neurons for richer wiring
        ncp_output = hidden_size // 2  # Motor neurons (projected up via proj_size)

        # Track the actual internal state size (= total wiring units)
        # This differs from hidden_size, which is the projected output size.
        self._state_size = ncp_units

        # AutoNCP automatically generates a biologically-inspired
        # sparse wiring pattern (sensory → inter → command → motor)
        wiring = AutoNCP(
            units=ncp_units,
            output_size=ncp_output,
        )

        # CfC: Closed-form Continuous-time RNN
        # - No ODE solver needed (closed-form solution)
        # - Supports variable time-steps via timespans
        # - proj_size projects the motor output back up to hidden_size
        self.cfc = CfC(
            input_size=input_size,
            units=wiring,
            proj_size=hidden_size,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        timespans: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the liquid cell.

        Args:
            x:         Input tensor of shape (batch, seq_len, input_size).
            hx:        Optional hidden state (batch, _state_size).
                       If None, initialized to zeros.
            timespans: Optional time deltas (batch, seq_len, 1).
                       Tells the cell how much real-time has passed
                       between each step. If None, uniform dt=1.0.

        Returns:
            output: Tensor of shape (batch, seq_len, hidden_size).
            hx_new: Updated hidden state (batch, _state_size).
        """
        output, hx_new = self.cfc(x, hx=hx, timespans=timespans)
        return output, hx_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a zero-initialized hidden state matching CfC internal size."""
        return torch.zeros(batch_size, self._state_size, device=device)


class LiquidStack(nn.Module):
    """
    A stack of multiple LiquidCell layers for deeper temporal processing.

    Each layer's output becomes the next layer's input, with residual
    connections to prevent gradient degradation.

    Args:
        input_size:  Dimension of the initial input.
        hidden_size: Dimension of each liquid cell.
        num_layers:  Number of stacked liquid cells.
        dropout:     Dropout between layers (not applied to last layer).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # First layer takes the raw input size
        self.layers = nn.ModuleList([
            LiquidCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        # Project input to hidden_size for residual connection on layer 0
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        hx_list: list[torch.Tensor] | None = None,
        timespans: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through all stacked liquid layers.

        Args:
            x:        Input (batch, seq_len, input_size).
            hx_list:  List of hidden states, one per layer.
            timespans: Time deltas (batch, seq_len, 1).

        Returns:
            output:      Final layer output (batch, seq_len, hidden_size).
            hx_list_new: Updated hidden states for each layer.
        """
        batch_size = x.size(0)
        device = x.device

        if hx_list is None:
            hx_list = [
                layer.init_hidden(batch_size, device)
                for layer in self.layers
            ]

        hx_list_new = []
        current = x

        for i, layer in enumerate(self.layers):
            # Store input for residual
            residual = self.input_proj(current) if i == 0 else current

            # Forward through liquid cell
            output, hx_new = layer(current, hx=hx_list[i], timespans=timespans)
            hx_list_new.append(hx_new)

            # Residual connection + LayerNorm + Dropout
            current = self.layer_norm(output + residual)
            if i < self.num_layers - 1:
                current = self.dropout(current)

        return current, hx_list_new
