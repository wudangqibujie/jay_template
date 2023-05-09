import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(BiLSTM, self).__init__()
        # input (seq_len, batch, input_dim)
        # h_0 (num_layers * num_directions, batch, hidden_size)
        # c_0 (num_layers * num_directions, batch, hidden_size)

        # output (seq_len, batch, hidden_size * num_directions)
        # h_n (num_layers * num_directions, batch, hidden_size)
        # c_n (num_layers * num_directions, batch, hidden_size)

        # input_dim, hidden_size, num_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        recu, _ = self.lstm(x)
        steps, batch_size, hidden_dim = recu.size()
        step_recu = recu.view(steps * batch_size, hidden_dim)
        out = self.embedding(step_recu) # [steps * batch_size, out_dim]
        out = out.view(steps, batch_size, -1)
        return out
