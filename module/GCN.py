import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """docstring for GCNLayer"""

    def __init__(self, channel_in, Ks, channel_out):
        super(GCNLayer, self).__init__()
        self.Ks = Ks + 1  # (0~Ks)
        self.channel_out = channel_out
        self.W = nn.Parameter(
            tc.zeros(size=(self.Ks * channel_in, channel_out)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # self.b = nn.Parameter(tc.zeros(size=(channel_out,)))
        # nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x, kernel):
        # x :       b*t*n*c
        # kernel:   n,n*Ks
        b, t, n, c = x.shape

        x = x.contiguous().view(-1, n, c).permute(0, 2, 1).view(-1, n)

        x_kernel = tc.mm(x, kernel).view(-1, c, self.Ks, n).permute(0,
                                                                    3, 1, 2).contiguous().view(-1, c * self.Ks)

        # F(H_next)  = acti(kernel*x*W)
        x_gconv = F.relu(tc.mm(x_kernel, self.W))  # + self.b)

        return x_gconv.view(b, t, n, self.channel_out)


class GCN(nn.Module):  # GCN with single layer
    """docstring for GCN"""

    def __init__(self, channel_in, Ks, channel_out):
        super(GCN, self).__init__()

        self.channel_in = channel_in
        self.Ks = Ks
        self.channel_out = channel_out

        self.W_first_order = nn.Parameter(
            tc.zeros(size=(self.channel_in, channel_out)))
        nn.init.xavier_uniform_(self.W_first_order.data, gain=1.414)

        self.gcnlayer = GCNLayer(self.channel_in, self.Ks, self.channel_out)

    def forward(self, x, A, L_cheb):
        b, t, n, c = x.shape

        if self.Ks == 1:  # first_order
            return F.relu(tc.matmul(tc.bmm(A.repeat(b * t, 1, 1), x.view(-1, n, c)), self.W_first_order))
        else:             # chey_poly
            out = self.gcnlayer(x, L_cheb)
            return F.relu(out)
