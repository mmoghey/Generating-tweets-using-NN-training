import torch
import torch.cuda
import torch.nn.functional as F
from torch.nn import Parameter


class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias_ih = True, bias_hh = False ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if bias_ih == True:
            self.b_ir = Parameter(torch.Tensor(hidden_size))
            self.b_in = Parameter(torch.Tensor(hidden_size))
            self.b_iz = Parameter(torch.Tensor(hidden_size))
        else :
            self.register_parameter('b_ir', None)
            self.register_parameter('b_in', None)
            self.register_parameter('b_iz', None)
            
        
        if bias_ih == True:
            self.b_hr = Parameter(torch.Tensor(hidden_size))
            self.b_hn = Parameter(torch.Tensor(hidden_size))
            self.b_hz = Parameter(torch.Tensor(hidden_size))
        else :
            self.register_parameter('b_hr', None)
            self.register_parameter('b_hn', None)
            self.register_parameter('b_hz', None)
          

        self.W_ir = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_in = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hn = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_iz = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hz = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

      

    def forward(self, inputs, hidden):
        """
        Perform a single timestep of a GRU cell using the provided input and the hidden state
        :param inputs: Current input
        :param hidden: Hidden state from the previous timestep
        :return: New hidden state
        """
        
        w_ir = F.linear(inputs, self.W_ir, self.b_ir)
        w_hr = F.linear(hidden, self.W_hr, self.b_hr)
        r = F.sigmoid(w_ir + w_hr)
        
        w_iz = F.linear(inputs, self.W_iz, self.b_iz)
        w_hz = F.linear(hidden, self.W_hz, self.b_hz)
        i = F.sigmoid(w_iz + w_hz)
        
        w_in = F.linear(inputs, self.W_in, self.b_in)
        w_hn = F.linear((r * hidden), self.W_hn, self.b_hn)
        n = F.tanh(w_in + w_hn)
        
        hidden_new = (1 - i) * n + i * hidden
        
        return hidden_new
