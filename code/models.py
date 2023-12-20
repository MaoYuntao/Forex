from transformer import TSTransformerEncoderClassiregressor
from torch import nn
import torch

#X: (Batch_size, Length, Feature_size)

class Transformer(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, n_heads, dropout, time_step):
        super().__init__()
        self.Model = TSTransformerEncoderClassiregressor(feat_dim=input_size, max_len=time_step, d_model=hidden_size,
                                                         n_heads=n_heads, num_layers=num_layers, dim_feedforward=hidden_size,
                                                         num_classes=1, dropout=dropout, pos_encoding='fixed',
                                                         activation='gelu', norm='BatchNorm')

    def forward(self, X):
        out = self.Model(X)
        return out


class CNN_Transformer(nn.Module):
    def __init__(self, stride, kernel_size, paddding, Transformer_layers,
                 input_size, hidden_size, n_heads, time_step, dropout=0.1):
        super().__init__()
        self.CNNs = nn.Sequential()
        self.timestep = time_step
        for i, (s, k, p) in enumerate(zip(stride, kernel_size, paddding)):
            self.CNNs.append(nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                       kernel_size=k, stride=s, padding=p))
            self.timestep = (self.timestep+2*p-k)//s+1

        self.Trans = TSTransformerEncoderClassiregressor(feat_dim=input_size, max_len=self.timestep, d_model=hidden_size,
                                                         n_heads=n_heads, num_layers=Transformer_layers, dim_feedforward=hidden_size,
                                                         num_classes=1, dropout=dropout, pos_encoding='fixed',
                                                         activation='gelu', norm='BatchNorm')

    def forward(self, X):
        X = X.permute(0, 2, 1)
        out = X
        for CNN in self.CNNs:
            out = CNN(out)
        out = out.permute(0, 2, 1)
        out = self.Trans(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input_seq):
        out, _ = self.lstm(input_seq.permute(1, 0, 2))
        out = self.linear(out[-1])
        return out


class CNN_LSTM(nn.Module):
    def __init__(self, stride, kernel_size, paddding, input_size, hidden_size, num_layers, time_step):
        super().__init__()
        self.CNNs = nn.Sequential()
        self.timestep = time_step
        for i, (s, k, p) in enumerate(zip(stride, kernel_size, paddding)):
            self.CNNs.append(nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                       kernel_size=k, stride=s, padding=p))
            self.timestep = (self.timestep + 2 * p - k) // s+1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, X):
        X = X.permute(0, 2, 1)
        out = X
        for CNN in self.CNNs:
            out = CNN(out)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out.permute(1, 0, 2))
        out = self.linear(out[-1])
        return out


#x = torch.rand((2, 30, 7))
#net = CNN_Transformer(input_size=7, hidden_size=128, kernel_size=(3, 2), stride=(3, 2),
#               paddding=(0, 0), time_step=30, n_heads=4, Transformer_layers=3)
#print(net(x))