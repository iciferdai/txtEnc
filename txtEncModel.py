from data_dict import *
from myTrans.enc_layer import *
from myTrans.pos import *

class TxtEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer() for _ in range(ENC_LAYER_NUM)
        ])
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_embedding = PosEncoding()
        self.fc = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x, mask=None):
        # 1.
        x = self.embedding(x) * math.sqrt(D_MODEL)
        x = self.pos_embedding(x)
        # 2.
        weights = []
        for layer in self.encoder_layers:
            x, w = layer(x, mask=mask)
            weights.append(w)
        o = self.fc(x)
        return o, x[:,0,:], weights