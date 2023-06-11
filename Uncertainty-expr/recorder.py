import torch
from torch import nn

class Recorder(nn.Module):
    
    def __init__(self, wrappedLayer, dropout_p = 0.5):
        super().__init__()
        self.wrappedLayer = wrappedLayer
        self.dropout = nn.Dropout(dropout_p)
        
        self.isRecord = False
        self.isReplay = False
        
    def switchToRecord(self):
        self.isReplay = False
        self.isRecord = True
        
    def switchToReplay(self):
        self.isReplay = True
        self.isRecord = False
        
    def switchToIdentity(self):
        self.isReplay = False
        self.isRecord = False
        
    def forward(self, x):
        
        if self.isRecord:
            z = self.wrappedLayer(x)
            self.buffer = z / x
            self.buffer[self.buffer != self.buffer] = 0 # Set NaN (x = 0) values to 0 (assume actfunc = 0 at x = 0)
            return z

        if self.isReplay:
            return self.dropout(self.buffer * x)
        
        return self.wrappedLayer(x)