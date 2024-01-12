import torch
from torch import nn


class NomalDecoder(nn.Module):
    def __init__(self, in_len: int, out_len: int, args):
        super().__init__()
        hidden_size = args.hidden_size
        self.fnn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, out_len),
            nn.Softmax(dim=1)
        )
        self.rnn = getattr(nn, args.encoder_cell)(input_size=in_len, hidden_size=hidden_size // 2,
                                              num_layers=args.num_layer, batch_first=True, bidirectional=True, dropout=0)

    def forward(self, x):
        x = self.rnn(x)[0]
        return self.fnn(x)

class HistoryDecoder(nn.Module):
    def __init__(self, in_len: int, out_len: int, args, encoder='GRU'):
        super().__init__()
        self.device = torch.device("cuda:%d" % args.device)
        self.history = None

        self.history_hidden = 128

        self.encoder = getattr(nn, encoder)(input_size=in_len, hidden_size=self.history_hidden // 2,
                                    num_layers=1, batch_first=True, bidirectional=True, dropout=0)

        self.rnn = getattr(nn, args.encoder_cell)(input_size=self.history_hidden, hidden_size=args.hidden_size // 2,
                                                  num_layers=args.num_layer, batch_first=True, bidirectional=True,
                                                  dropout=0)
        self.fnn = nn.Sequential(
            nn.Linear(args.hidden_size, out_len),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)[0]
        if self.history != None:
            x1 = torch.cat((self.history, x1), dim=0)
        else:
            self.history = x1
        x1 = self.rnn(x1)[0]
        x1 = self.fnn(x1)[-x.shape[0]:, :]
        return x1

    def reset(self):
        self.history = None