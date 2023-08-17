import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



# credits for blog https://cnvrg.io/pytorch-lstm/

class LSTM_Model(nn.Module):

    def __init__(self, input_size,
                 layers_sizes,
                 num_layers,
                 num_classes,
                 dropout,
                 activation, 
                 bidirectional=True,
                 bias=True):
        
        super(LSTM_Model, self).__init__()
        self.input_size = input_size # number of expected features in the input x
        self.layers_sizes = layers_sizes # sizes of layers
        self.num_layers = num_layers # number of recurrent layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional # if True, becomes a bidirectional LSTM
        self.bias = bias # if False, then the layer does not use bias weights b_ih and b_hh
        self.activation = activation

        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.layers_sizes[0],
                                  num_layers=self.num_layers,
                                  batch_first=True,
                                  bidirectional=bidirectional,
                                  dropout=dropout,
                                  bias=bias)

        
        
        # organize layer sizes, inputs and outputs
        self.layers = nn.ModuleList()
        # if bidirectional, take into account doubled layer sizes
        layer1 = 2*self.layers_sizes[0] if self.bidirectional else self.layers_sizes[0]
        input_size = layer1
        for size in self.layers_sizes[1:]:
            self.layers.append(torch.nn.Linear(input_size, size))
            input_size = size
        self.output_layer = torch.nn.Linear(self.layers_sizes[-1], self.num_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout) 


    def forward(self, features, lengths):

        padded_input = pack_padded_sequence(features,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)

        if self.bidirectional: # if bidirectional, take into account doubled layer sizes
            nl = 2*self.num_layers
        else:
            nl = self.num_layers
        h0 = torch.zeros(nl, features.size(0), self.layers_sizes[0]).to(features.device)
        c0 = torch.zeros(nl, features.size(0), self.layers_sizes[0]).to(features.device)
        
        padded_output, _ = self.lstm(padded_input, (h0, c0))
        output, input_sizes = pad_packed_sequence(padded_output, batch_first=True)

        output = torch.stack([el[lengths[i] - 1] for i, el in enumerate(output)])
        lstm_activation = self.activation(output)
        for i, linear_layer in enumerate(self.layers):
            input_linear = lstm_activation
            input_linear = linear_layer(self.dropout(input_linear))
            lstm_activation = self.activation(input_linear)
        output = self.output_layer(lstm_activation)
        
        return self.softmax(output)



class LSTM_Model_BN(nn.Module):

    def __init__(self, input_size,
                 layers_sizes,
                 num_layers,
                 num_classes,
                 dropout,
                 activation, 
                 bidirectional=True,
                 bias=True):
        
        super(LSTM_Model_BN, self).__init__()
        self.input_size = input_size # number of expected features in the input x
        self.layers_sizes = layers_sizes # sizes of layers
        self.num_layers = num_layers # number of recurrent layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional # if True, becomes a bidirectional LSTM
        self.bias = bias # if False, then the layer does not use bias weights b_ih and b_hh
        self.activation = activation

        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.layers_sizes[0],
                                  num_layers=self.num_layers,
                                  batch_first=True,
                                  bidirectional=bidirectional,
                                  dropout=dropout,
                                  bias=bias)

        self.layers = nn.ModuleList()
        # if bidirectional, take into account doubled layer sizes
        layer1 = 2*self.layers_sizes[0] if self.bidirectional else self.layers_sizes[0]
        input_size = layer1
        for size in self.layers_sizes[1:]:
            self.layers.append(torch.nn.Linear(input_size, size))
            input_size = size
        self.output_layer = torch.nn.Linear(self.layers_sizes[-1], self.num_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(torch.nn.BatchNorm1d(layer1))
        for size in self.layers_sizes[1:]:
            self.batch_norms.append(torch.nn.BatchNorm1d(size))
        self.batch_norms.append(torch.nn.BatchNorm1d(self.num_classes))
        
        self.dropout = nn.Dropout(p=dropout) 


    def forward(self, features, lengths):

        padded_input = pack_padded_sequence(features,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)

        if self.bidirectional: # if bidirectional, take into account doubled layer sizes
            nl = 2*self.num_layers
        else:
            nl = self.num_layers
        h0 = torch.zeros(nl, features.size(0), self.layers_sizes[0]).to(features.device)
        c0 = torch.zeros(nl, features.size(0), self.layers_sizes[0]).to(features.device)
        
        padded_output, _ = self.lstm(padded_input, (h0, c0))
        output, input_sizes = pad_packed_sequence(padded_output, batch_first=True)

        output = torch.stack([el[lengths[i] - 1] for i, el in enumerate(output)])
        first_batch_norm = self.batch_norms[0]
        output = first_batch_norm(output)
            
        lstm_activation = self.activation(output)
        for i, linear_layer in enumerate(self.layers):
            input_linear = lstm_activation
            input_linear = linear_layer(self.dropout(input_linear))
            batch_norm = self.batch_norms[i+1]
            input_linear = batch_norm(input_linear)
            lstm_activation = self.activation(input_linear)
        output = self.output_layer(lstm_activation)
        last_batch_norm = self.batch_norms[-1]
        output = last_batch_norm(output)
        return self.softmax(output)