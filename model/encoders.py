import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MAP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, 
                 dropout=0.2, device=DEVICE, lr=3e-5):

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers, 
                            dropout=dropout,
                            bidirectional=True).to(device)
        
        self.entity_mapper = nn.Linear(hidden_size*lstm_layers, output_size).to(device)
        # self.relation_mapper = nn.Linear(hidden_size*lstm_layers, output_size).to(device)
        

    # def forward(self, x: torch.Tensor, entity_masks: torch.Tensor, relation_masks: torch.Tensor):
    def forward(self, x: torch.Tensor, batch_first=True):
        lstm_out, _ = self.lstm(x)
        unpacked_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        entity_hidden = self.combine_entity_states(unpacked_hidden, spans)
        scores = self.entity_mapper(entity_hidden)

        return scores  


class MAP:
    pass