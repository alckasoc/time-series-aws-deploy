import torch
from torch import nn
from torch.nn import functional as F

# Ref: https://www.kaggle.com/code/sagarikajadon/gb-vpp-pytorch-lstm-baseline
class BiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        hidden_dim= [400, 300, 200, 100]
        self.bilstm1= nn.LSTM(input_dim, hidden_dim[0], batch_first= True, bidirectional= True)
        self.norm1= nn.LayerNorm(hidden_dim[0]*2)
        
        self.bilstm2= nn.LSTM(hidden_dim[0]*2, hidden_dim[1], batch_first= True, bidirectional= True)
        self.norm2= nn.LayerNorm(hidden_dim[1]*2)
        
        self.bilstm3= nn.LSTM(hidden_dim[1]*2, hidden_dim[2], batch_first= True, bidirectional= True)
        self.norm3= nn.LayerNorm(hidden_dim[2]*2)
        
        self.bilstm4= nn.LSTM(hidden_dim[2]*2, hidden_dim[3], batch_first= True, bidirectional= True)
        self.norm4= nn.LayerNorm(hidden_dim[3]*2)
        
        self.fc1= nn.Linear(hidden_dim[3]*2, 100)
        self.fc2= nn.Linear(100, output_dim)

        
    def forward(self, X):
        pred, _= self.bilstm1(X)
        pred= self.norm1(pred)
        
        pred, _= self.bilstm2(pred)
        pred= self.norm2(pred)
        
        pred, _= self.bilstm3(pred)
        pred= self.norm3(pred)
        
        pred, _= self.bilstm4(pred)
        pred= self.norm4(pred)
        
        pred= self.fc1(pred[:, -1, :])
        pred= F.selu(pred)
        
        pred= self.fc2(pred)
        return pred