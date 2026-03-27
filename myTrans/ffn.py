from myTrans.base_params import *

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        o = self.fc2(x)
        return o