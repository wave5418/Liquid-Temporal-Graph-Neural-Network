from torch import nn
import torch
from ncps.torch import LTC,CfC,LTCCell,CfCCell
from ncps.wirings import AutoNCP
import ncps


class cfcbundle(nn.Module):
    def __init__(self, input_size, hidden_size, mode="default", backbone_layers=1):
        super().__init__()
        self.batch_first = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.backbone_layers = backbone_layers
        self.rnn_cell = CfCCell(input_size=self.input_size, hidden_size=self.hidden_size, mode=self.mode,
                                backbone_layers=self.backbone_layers, backbone_units=128, backbone_dropout=0.2)
        self.leaky_factor = torch.nn.Linear(input_size, hidden_size)

    def forward(self, input, hx, timespans=None):
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)
        memory_embed = self.leaky_factor(input)
        memory_embed = memory_embed.squeeze(1)
        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
        else:
            h_state = hx
        if is_batched:
            if h_state.dim() != 2:
                msg = (
                    "For batched 2-D input, hx and cx should "
                    f"also be 2-D but got ({h_state.dim()}-D) tensor"
                )
                raise RuntimeError(msg)
        else:
            # batchless  mode
            if h_state.dim() != 1:
                msg = (
                    "For unbatched 1-D input, hx and cx should "
                    f"also be 1-D but got ({h_state.dim()}-D) tensor"
                )
                raise RuntimeError(msg)
            h_state = h_state.unsqueeze(0)
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            # print(h_state.shape,memory_embed.shape)
            h_out, h_state = self.rnn_cell.forward(inputs, h_state + memory_embed, ts)

        #    print(hx,self.leaky_factor)
        hx = h_state

        return hx, hx