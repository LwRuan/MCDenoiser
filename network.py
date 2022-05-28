from common import envs, constants, switches, nnconfigs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def MakeNN(n_layers, mode):
  layers = [
    nn.Conv2d(nnconfigs.input_channels, nnconfigs.hidden_channels, nnconfigs.kernel_size),
    nn.ReLU()
  ]
  for l in range(n_layers - 2):
    layers += [
      nn.Conv2d(nnconfigs.hidden_channels, nnconfigs.hidden_channels, nnconfigs.kernel_size),
      nn.ReLU()
    ]
  out_channels = 3 if mode == 'DPCN' else constants.output_kernel_size**2
  layers += [nn.Conv2d(nnconfigs.hidden_channels, out_channels, nnconfigs.kernel_size)]
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  return nn.Sequential(*layers)

if __name__ == "__main__":
  MakeNN(nnconfigs.L, switches.mode)
  
