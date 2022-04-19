import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_dim, output_dim)

        self.layers = [self.l1, self.a1,
                       self.l2, self.a2,
                       self.l3, self.a3,
                       self.l4]
        """
        self.layers = [self.l1, self.a1,
                        self.l2, self.a2,                   
                       self.l4]
        """
                       
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
