from common import envs, constants, switches, nnconfigs, trainconfigs
from network import MakeNN
from dataset import KPCNDataset, ToTorchTensors
from process import TestDataFromFiles, PostprocessColor
from train import CropLike, ApplyKernel, Crop
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
  permutation = [0, 3, 1, 2]
  inv_permute = [0, 2, 3, 1]
  for i, scene in enumerate(test_data):
    data = scene["data"][np.newaxis,...].permute(permutation).to(device)
    gt = scene["gt"][np.newaxis,...].permute(permutation).to(device)
    output = denoiser(data)
    h, w = output.shape[-2], output.shape[-1]
    if switches.mode == "KPCN":
      # X_input = CropLike(X, output)
      output = ApplyKernel(output, data, device)
      output = Crop(output, h, w)
    noisy = CropLike(data, output)
    with torch.no_grad():
      gt = CropLike(gt, output).cpu().permute(inv_permute)
      gt_color = np.clip(PostprocessColor(gt[...,:3],gt[...,3:6]) * 255.0, 0, 255.0)
      noisy = noisy.cpu().permute(inv_permute)
      noisy_color = np.clip(PostprocessColor(noisy[...,:3], noisy[...,3:6]) * 255.0, 0, 255.0)
      output = output.cpu().permute(inv_permute)
      output_color = np.clip(PostprocessColor(output[...,:3], noisy[...,3:6]) * 255.0, 0, 255.0)
      imageio.imsave(f"tests/{mode}/{i}-noisy.png", noisy_color[0].type(torch.uint8))
      imageio.imsave(f"tests/{mode}/{i}-denoised.png", output_color[0].type(torch.uint8))
      imageio.imsave(f"tests/{mode}/{i}-gt.png", gt_color[0].type(torch.uint8))
    
    
if __name__ == "__main__":
  # Test("./models/dpcn-500.pth", "DPCN")
  Test("./models/KPCN-2000.pth", "KPCN")

  # test_data = torch.load("./data/cropped.pt")
  # print(test_data[2000]["data"][...,:3].max())
  # noisy_color = np.clip(test_data[2000]["data"][...,:3] * 255.0, 0, 255.0)
  # print(test_data[2000]["gt"][...,:3].max())
  # gt_color = np.clip(test_data[2000]["gt"][...,:3] * 255.0, 0, 255.0)
  # imageio.imsave(f"imgs/image-noisy.png", noisy_color.type(torch.uint8))
  # imageio.imsave(f"imgs/image-gt.png", gt_color.type(torch.uint8))

  # gt = test_data[2000]["gt"][np.newaxis,...,:3]
  # gt_color = np.clip(gt * 255.0, 0, 255.0)
  # ws = np.zeros((1,9,gt.shape[1],gt.shape[2]))
  # for i, v in enumerate([1.,2.,1.,0.,0.,0.,-1.,-2.,-1.]):
  #   ws[0,i] = np.ones((gt.shape[1],gt.shape[2]))
  # print(ws.shape)
  # device = torch.device("cpu")
  # out = ApplyKernel(torch.from_numpy(ws), gt.permute((0,3,1,2)), device)
  # out = out.permute(0,2,3,1)
  # out_color = np.clip(out * 255.0, 0, 255.0)
  # imageio.imsave(f"imgs/test1.png", gt_color[0].type(torch.uint8))
  # imageio.imsave(f"imgs/test2.png", out_color[0].type(torch.uint8))
  

