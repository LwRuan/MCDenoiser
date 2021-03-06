from common import envs, constants, switches, nnconfigs, trainconfigs
from network import MakeNN
from dataset import KPCNDataset
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def CropLike(data, like, debug=False):
  if data.shape[-2:] != like.shape[-2:]:
    with torch.no_grad():
      dx, dy = data.shape[-2] - like.shape[-2], data.shape[-1] - like.shape[-1]
      data = data[...,dx//2:-dx//2,dy//2:-dy//2]
      if debug:
        print(dx, dy)
        print("After crop:", data.shape)
  return data

def Crop(data, h, w):
  if data.shape[-2:] != (h, w):
    with torch.no_grad():
      dx, dy = data.shape[-2] - h, data.shape[-1] - w
      data = data[...,dx//2:-dx//2,dy//2:-dy//2]
  return data

def ApplyKernel(weights, data, device):
  weights = weights.permute((0, 2, 3, 1)).to(device)
  _, h, w, _ = weights.size()
  weights = F.softmax(weights, dim=3).view(-1, w * h, constants.output_kernel_size, constants.output_kernel_size)
  r = constants.output_kernel_size // 2
  p = (data.size()[-1] - w) // 2
  R = []
  G = []
  B = []
  kernels = []
  for i in range(h):
    for j in range(w):
      pos = i*h+j
      ws = weights[:,pos:pos+1,:,:]
      kernels += [ws, ws, ws]
      sy, ey = i+p-r, i+p+r+1
      sx, ex = j+p-r, j+p+r+1
      R.append(data[:,0:1,sy:ey,sx:ex])
      G.append(data[:,1:2,sy:ey,sx:ex])
      B.append(data[:,2:3,sy:ey,sx:ex])
  reds = (torch.cat(R, dim=1).to(device)*weights).sum(2).sum(2)
  greens = (torch.cat(G, dim=1).to(device)*weights).sum(2).sum(2)
  blues = (torch.cat(B, dim=1).to(device)*weights).sum(2).sum(2)
  res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(device)
  return res

def Train(start=0, model_path="./models/KPCN-200.pth"):
  writer = SummaryWriter('logs')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  # load data
  cropped = torch.load("./data/cropped.pt")
  dataset = KPCNDataset(cropped)
  print(f"data loaded, {len(dataset)} patches")
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
  denoiser = MakeNN(nnconfigs.L, switches.mode).to(device)
  criterion = nn.L1Loss()
  print(denoiser, "CUDA:", next(denoiser.parameters()).is_cuda)
  optimizer = optim.Adam(denoiser.parameters(), lr=trainconfigs.learning_rate)
  start_time = time.time()
  permutation = [0, 3, 1, 2]
  if start != 0: denoiser.load_state_dict(torch.load(model_path))
  for epoch in range(start, trainconfigs.epochs):
    loss_batch = 0
    cnt = 0
    for i_batch, sample_batch in enumerate(dataloader):
      cnt += 1
      X = sample_batch["data"].permute(permutation).to(device)
      Y = sample_batch["gt"][:,:,:,:3].permute(permutation).to(device)
      optimizer.zero_grad()
      output = denoiser(X)
      h, w = output.shape[-2], output.shape[-1]
      if switches.mode == "KPCN":
        # X_input = CropLike(X, output)
        output = ApplyKernel(output, X, device)
        output = Crop(output, h, w)
      Y = CropLike(Y, output)
      loss = criterion(output, Y)
      loss.backward()
      optimizer.step()
      loss_batch += loss.item()
    loss_batch /= cnt
    print(f"epoch: {epoch}, loss: {loss_batch}")
    writer.add_scalar("loss", loss_batch, epoch)
    if (epoch + 1)%100 == 0:
      torch.save(denoiser.state_dict(), f"models/{switches.mode}-{epoch+1}.pth")

if __name__ == "__main__":
  Train()
