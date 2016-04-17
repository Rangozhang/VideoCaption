require 'nn'
require 'nngraph'
LSTM = require 'LSTM'

m = LSTM.VC_lstm(10, 5, 3, 2, 8, 2, 0)
print(m)

x = torch.randn(2, 10)
print(m:forward{x, torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 3)})
