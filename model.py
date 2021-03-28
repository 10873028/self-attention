import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention, self).__init__()
        self.q = nn.Linear(in_features, out_features, bias=None)
        self.k = nn.Linear(in_features, out_features, bias=None)
        self.v = nn.Linear(in_features, out_features, bias=None)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: N, 1
        q = self.q(x).view(-1, 1)
        k = self.k(x)
        v = self.v(x)
        alpha = q * k
        alpha = self.softmax(alpha)
        out = torch.inner(v, alpha)
        return out

if __name__ == '__main__':
    a = torch.FloatTensor([1,2,3,4])
    b = torch.FloatTensor([1,2,3,4])
    model = Attention(4, 4)
    print(model(a))

