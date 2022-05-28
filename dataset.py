import torch
import numpy as np
from process import CroppedFromFiles

def ToTorchTensors(data):
  if isinstance(data, dict):
    for k, v in data.items():
      if not isinstance(v, torch.Tensor):
        data[k] = torch.from_numpy(v)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if not isinstance(v, torch.Tensor):
        data[i] = ToTorchTensors(v)
  return data
 
def SendToDevice(data, device):
  if isinstance(data, dict):
    for k, v in data.items():
      if isinstance(v, torch.Tensor):
        data[k] = v.to(device)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if isinstance(v, torch.Tensor):
        data[i] = v.to(device)
  return data

class KPCNDataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.samples = data

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]

if __name__ == "__main__":
  cropped = CroppedFromFiles()
  cropped = ToTorchTensors(cropped)
  torch.save(cropped, "./data/cropped.pt")
  # cropped = torch.load("./data/cropped.pt")
  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # print(f"device: {device}")
  # cropped = SendToDevice(cropped, device)
  # dataset = KPCNDataset(cropped)
  # print(f"dataset size: {len(dataset)}")

