import torch.nn as nn

class SoftmaxRegressionModel(nn.Module):
    
    def __init__(self):

        super().__init__()        
        self.fc1 = nn.Linear(52,13)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(13,3)

    def forward(self,x):

        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
