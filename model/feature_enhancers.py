import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
import multiprocessing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureEnhancer(nn.Module, ABC):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @abstractmethod
    def forward(self, x):
        pass


class Pass(FeatureEnhancer):

    def __init__(self, device=DEVICE, *args, **kwargs):
        super().__init__(device) 

    def prepare_input(self, x, *args):
        return x
        
    def forward(self, x):
        return x

    def __repr__(self):
        return "Pass()"

    def __str__(self):
        return "Pass" 

    
class MAP(FeatureEnhancer):

    def __init__(self, input_size, output_size, 
                 dropout=0.2, device=DEVICE, lr=1e-3):
        super().__init__(device) 
        modules = [
            ('HiddenOut', nn.Linear(input_size, output_size)),
        ]
        self.model = nn.Sequential(OrderedDict(modules)).to(device) 

    def prepare_input(self, x, *args):
        return x.reshape(-1, x.shape[-1])
        
    def prepare_output(self, h, orig_shape):
        return h.reshape(orig_shape) 

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return "MAP()"

    def __str__(self):
        return "MAP" 

class MLP(FeatureEnhancer):

    def __init__(self, input_size, output_size,
        hidden_size=768, dropout=0.2, device=DEVICE, lr=1e-3):
        super().__init__(device) 
        modules = [
            ('InputMap', nn.Linear(input_size, input_size)),
            ('Dropout1', nn.Dropout(dropout)),
            ('Tanh', nn.Tanh()),
            ('Hidden1', nn.Linear(input_size, hidden_size)),
            ('Dropout2', nn.Dropout(dropout)),
            ('ReLU', nn.ReLU()),
            ('HiddenOut', nn.Linear(hidden_size, output_size)),
        ]
        self.model = nn.Sequential(OrderedDict(modules)).to(device) 

    def prepare_input(self, x, *args):
        return x.reshape(-1, x.shape[-1])

    def prepare_output(self, h, orig_shape):
        return h.reshape(orig_shape) 

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return "MLP()"

    def __str__(self):
        return "MLP" 

        
class BiLSTM(FeatureEnhancer):
    def __init__(self, input_size, output_size, lstm_layers=2,
       dropout=0.2, device=DEVICE, lr=3e-5, batch_first=True):
        super().__init__(device) 

        # output size has to be divisible by two for bidirectional
        assert(output_size%2==0)
        hidden_size = int(output_size/2)

        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout).to(device)
        self.model = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers, 
                            dropout = dropout,
                            bidirectional=True,
                            batch_first=batch_first).to(device)

    def forward(self, x):
        lstm_out, _ = self.model(x)
        unpacked_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=self.batch_first)

        return unpacked_hidden 


    def prepare_input(self, x, mask):
        sequence_dimension = 1 if self.batch_first else 0
        lengths = mask.sum(dim=sequence_dimension).int().tolist()
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=self.batch_first, 
                                                           enforce_sorted=False)
        return packed_x


    def prepare_output(self, h, orig_shape):
        h_orig = torch.zeros(orig_shape, device=self.device)
        if self.batch_first:
            h_orig[:, :h.shape[1], :] = h
        else:
            h_orig[:h.shape[0], :, :] = h

        return h_orig    


    def __repr__(self):
        return "BiLSTM()"

    def __str__(self):
        return "BiLSTM"  


_FEATURE_ENHANCERS = {
    'pass': Pass,
    'map': MAP,
    'mlp': MLP,
    'bilstm': BiLSTM
}

def get_feature_enhancer(name):
    return _FEATURE_ENHANCERS[name]
