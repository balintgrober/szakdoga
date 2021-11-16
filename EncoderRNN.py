import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, h_hidden, c_hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hiddens = self.lstm(output, (h_hidden, c_hidden))
        return output, hiddens

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device="cuda")