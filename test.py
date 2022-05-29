from common import envs, constants, switches, nnconfigs, trainconfigs
from network import MakeNN
from dataset import KPCNDataset, ToTorchTensors
from process import TestDataFromFiles, PostprocessColor
from train import CropLike, ApplyKernel
import numpy as np
import imageio.v2  as imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def Test(model_path, mode="DPCN"):
  test_data = ToTorchTensors(TestDataFromFiles())
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  denoiser = MakeNN(nnconfigs.L, mode).to(device)
  denoiser.load_state_dict(torch.load(model_path))
  permutation = [2, 0, 1]
  inv_permute = [1, 2, 0]
  for i, scene in enumerate(test_data):
    data = scene["data"].permute(permutation).to(device)
    gt = scene["gt"].permute(permutation).to(device)
    output = denoiser(data)
    noisy = CropLike(data, output)
    if mode == "KPCN":
      output = ApplyKernel(output, noisy)
    with torch.no_grad():
      gt = CropLike(gt, output).cpu().permute(inv_permute)
      gt_color = np.clip(PostprocessColor(gt[...,:3],gt[...,3:6]) * 255.0, 0, 255.0)
      noisy = noisy.cpu().permute(inv_permute)
      noisy_color = np.clip(PostprocessColor(noisy[...,:3], noisy[...,3:6]) * 255.0, 0, 255.0)
      output = output.cpu().permute(inv_permute)
      output_color = np.clip(PostprocessColor(output[...,:3], noisy[...,3:6]) * 255.0, 0, 255.0)
      imageio.imsave(f"tests/{i}-noisy.png", noisy_color.type(torch.uint8))
      imageio.imsave(f"tests/{i}-denoised.png", output_color.type(torch.uint8))
      imageio.imsave(f"tests/{i}-gt.png", gt_color.type(torch.uint8))
    
    
if __name__ == "__main__":
  Test("./models/dpcn.pth", "DPCN")
