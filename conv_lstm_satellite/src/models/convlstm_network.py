import torch
import torch.nn as nn

class ConvLSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvLSTMNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.convlstm_cells = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convlstm_cells.append(ConvLSTMCell(in_channels, hidden_dim, kernel_size))

        self.fc = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        hidden_states = [None] * self.num_layers

        for t in range(seq_len):
            for i in range(self.num_layers):
                # input to layer 0 is the raw input; for others use previous layer's hidden state
                layer_input = x[:, t, :, :, :] if i == 0 else hidden_states[i-1][0]
                hidden_states[i] = self.convlstm_cells[i](layer_input, hidden_states[i])

        output = self.fc(hidden_states[-1][0])
        return output

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x, hidden_state):
        if hidden_state is None:
            h_t, c_t = (torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3)).to(x.device),
                        torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3)).to(x.device))
        else:
            h_t, c_t = hidden_state

        combined = torch.cat((x, h_t), dim=1)
        conv_output = self.conv(combined)
        
        i_t, f_t, o_t, g_t = conv_output.chunk(4, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t