import torch
import torch.nn as nn


class Mixtape(nn.Module):
    def __init__(self, query_dim, mey_dim, mix_dim, M=4):
        super(Mixtape, self).__init__()
        self.M = M
        self.input_dim = input_dim
        self.mix_dim = mix_dim
        self.tanh = nn.Tanh(dim=-1)
        self.H_m = []
        self.U_m = []
        for i in range(M):
            H_m = nn.Linear(query_dim, mey_dim, bias=False)
            U_m = nn.Linear(query_dim, mix_dim, bias=False)
            self.H_m.append(H_m)
            self.U_m.append(U_m)
        self.H_m = nn.ModuleList(self.H_m)
        self.U_m = nn.ModuleList(self.U_m)


    def forward(query, mey):
        """
            INPUT:  query, [bz, length, query_dim]
                    mey, [bz, length, mey_dim]
        """
        for i in range(self.M):
            # [bz, length, mey_dim]
            h_qm = self.tanh(self.H_m[i](query))
            # l_qkm = 







        