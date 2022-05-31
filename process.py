from pip import main
import imageio.v2 as imageio
from common import constants, envs, switches
import numpy as np
import PIL
from scipy import ndimage
import random
from random import randint
import glob

def PreprocessVariance(variance):
  variance = variance.max(axis=2)
  return variance

def PreprocessColor(color, albedo):
  return color.astype(np.float32) / (albedo + constants.eps)

def PostprocessColor(color, albedo):
  return color * (albedo + constants.eps)

def PreprocessColorVar(variance, albedo):
  variance = variance / (albedo + constants.eps)**2
  return PreprocessVariance(variance)

def Gradients(image):
  y, x, c = image.shape
  dx = image[:, 1:, :] - image[:, :x-1, :]
  dy = image[1:, :, :] - image[:y-1, :, :]
  dx = np.append(dx, np.zeros([y, 1, c]), axis=1)
  dy = np.append(dy, np.zeros([1, x, c]), axis=0)
  grad = np.dstack((dx, dy))
  return grad

def Make3Channels(image):
    if image.ndim == 2: image = np.dstack((image, image, image))
    if image.shape[2] == 1: 
        image = np.reshape(image, (image.shape[0], image.shape[1]))
        image = np.dstack((image, image, image))
    image = image[:,:,:3]
    return image

def LoadHighSpp(folder_path):
  albedo = Make3Channels(imageio.imread(folder_path + "/albedo.png").astype(np.float32)) / 255.0
  color = Make3Channels(imageio.imread(folder_path + "/final.png").astype(np.float32)) / 255.0
  if switches.albedo_div:
    color = PreprocessColor(color, albedo)
  
  combined = np.dstack((color, albedo))
  # combined = combined.astype(np.float16)
  assert not np.any(np.isnan(combined))
  return combined

def LoadLowSpp(folder_path, return_channels = False):
  # albedo
  albedo = Make3Channels(imageio.imread(folder_path + "/albedo.png").astype(np.float32)) / 255.0
  albedo_variance = Make3Channels(imageio.imread(folder_path + "/albedoVariance.png").astype(np.float32)) / 255.0
  albedo_variance = PreprocessVariance(albedo_variance)
  albedo_variance = albedo_variance.reshape((albedo_variance.shape[0], albedo_variance.shape[1], 1)) / 255.0
  albedo_gradient = Gradients(albedo)

  # color
  color = Make3Channels(imageio.imread(folder_path + "/final.png").astype(np.float32)) / 255.0
  if switches.albedo_div: color = PreprocessColor(color, albedo)
  color_variance = Make3Channels(imageio.imread(folder_path + "/colorVariance.png")).astype(np.float32) / 255.0
  if switches.albedo_div:
    color_variance = PreprocessColorVar(color_variance, albedo)
  else:
    color_variance = PreprocessVariance(color_variance)
  color_variance = color_variance.reshape((color_variance.shape[0], color_variance.shape[1], 1))
  color_gradient = Gradients(color)
  
  # depth
  depth = imageio.imread(folder_path + "/depth.png").astype(np.float32) / 255.0
  depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
  depth_variance = Make3Channels(imageio.imread(folder_path + "/depthVariance.png").astype(np.float32)) / 255.0
  depth_variance = PreprocessVariance(depth_variance)
  depth_variance = depth_variance.reshape((depth_variance.shape[0], depth_variance.shape[1], 1))

  # normal
  normal = Make3Channels(imageio.imread(folder_path + "/normal.png").astype(np.float32)) / 255.0
  normal_variance = Make3Channels(imageio.imread(folder_path + "/normalVariance.png").astype(np.float32)) / 255.0
  normal_variance = PreprocessVariance(normal_variance)
  normal_variance = normal_variance.reshape((normal_variance.shape[0], normal_variance.shape[1], 1))

  chans = (color, albedo, color_gradient, albedo_gradient, depth, normal, color_variance, albedo_variance, depth_variance, normal_variance)
  chans_str = "color, albedo, color_gradient, albedo_gradient, depth, normal, color_variance, albedo_variance, depth_variance, normal_variance".split(", ")
  names_and_channels = []
  if return_channels:
    for i in range(len((chans))):
      names_and_channels.append((chans_str[i], chans[i].shape[2]))
  
  combined = np.dstack((color, albedo, color_gradient, albedo_gradient, depth, normal, color_variance, albedo_variance, depth_variance, normal_variance))
  #  combined = combined.astype(np.float16)
  assert not np.any(np.isnan(combined))

  if return_channels:
    return combined, names_and_channels
    
  return combined

def ImShow(data):
  data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
  pil_img = PIL.Image.fromarray(data, 'RGB')
  return pil_img

def ImShowPP(data):
  if switches.albedo_div:
    data = PostprocessColor(data[:,:,:3], data[:,:,3:6])
  return ImShow(data[:,:,:3])


# for patches
def GetVarianceMap(data, patch_size, relative=False):
  # introduce a dummy third dimension if needed
  if data.ndim < 3: data = data[:,:,np.newaxis]
  # compute variance
  mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
  sqrmean = ndimage.uniform_filter(data**2, size=(patch_size, patch_size, 1))
  variance = np.maximum(sqrmean - mean**2, 0)
  # convert to relative variance if requested
  if relative:
      variance = variance/np.maximum(mean**2, 1e-2)
  # take the max variance along the three channels, gamma correct it to get a
  # less peaky map, and normalize it to the range [0,1]
  variance = variance.max(axis=2)
  variance = np.minimum(variance**(1.0/2.2), 2.0)
  # variance = variance**(1.0/2.2)
  return variance/variance.max()

# Generate importance sampling map based on buffer and desired metric
def GetImportanceMap(buffers, metrics, weights, patch_size):
  if len(metrics) != len(buffers):
    metrics = [metrics[0]]*len(buffers)
  if len(weights) != len(buffers):
    weights = [weights[0]]*len(buffers)
  impMap = None
  for buf, metric, weight in zip(buffers, metrics, weights):
    if metric == 'uniform':
      cur = np.ones(buf.shape[:2], dtype=np.float)
    elif metric == 'variance':
      cur = GetVarianceMap(buf, patch_size, relative=False)
    elif metric == 'relvar':
      cur = GetVarianceMap(buf, patch_size, relative=True)
    else:
      print('Unexpected metric:', metric)
    if impMap is None:
      impMap = cur*weight
    else:
      impMap += cur*weight
  return impMap / impMap.max()


# Sample patches using dart throwing (works well for sparse/non-overlapping patches)
def SamplePatchesProg(img_dim, patch_size, n_samples, maxiter=5000):
  # estimate each sample patch area
  full_area = float(img_dim[0]*img_dim[1])
  sample_area = full_area/n_samples
  # get corresponding dart throwing radius
  radius = np.sqrt(sample_area/np.pi)
  minsqrdist = (2*radius)**2
  # compute the distance to the closest patch
  def get_sqrdist(x, y, patches):
    if len(patches) == 0:
      return np.infty
    dist = patches - [x, y]
    return np.sum(dist**2, axis=1).min()
  # perform dart throwing, progressively reducing the radius
  rate = 0.96
  patches = np.zeros((n_samples,2), dtype=int)
  xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
  ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
  for patch in range(n_samples):
    done = False
    while not done:
      for i in range(maxiter):
        x = randint(xmin, xmax)
        y = randint(ymin, ymax)
        sqrdist = get_sqrdist(x, y, patches[:patch,:])
        if sqrdist > minsqrdist:
          patches[patch,:] = [x, y]
          done = True
          break
      if not done:
        radius *= rate
        minsqrdist = (2*radius)**2
  return patches

def PrunePatches(shape, patches, patchsize, imp):
  pruned = np.empty_like(patches)
  # Generate a set of regions tiling the image using snake ordering.
  def get_regions_list(shape, step):
    regions = []
    for y in range(0, shape[0], step):
      if y//step % 2 == 0:
        xrange = range(0, shape[1], step)
      else:
        xrange = reversed(range(0, shape[1], step))
      for x in xrange:
        regions.append((x, x + step, y, y + step))
    return regions
  # Split 'patches' in current and remaining sets, where 'cur' holds the
  # patches in the requested region, and 'rem' holds the remaining patches.
  def split_patches(patches, region):
    cur = np.empty_like(patches)
    rem = np.empty_like(patches)
    ccount, rcount = 0, 0
    for i in range(patches.shape[0]):
      x, y = patches[i,0], patches[i,1]
      if region[0] <= x < region[1] and region[2] <= y < region[3]:
        cur[ccount,:] = [x,y]
        ccount += 1
      else:
        rem[rcount,:] = [x,y]
        rcount += 1
    return cur[:ccount,:], rem[:rcount,:]
  # Process all patches, region by region, pruning them randomly according to
  # their importance value, ie. patches with low importance have a higher
  # chance of getting pruned. To offset the impact of the binary pruning
  # decision, we propagate the discretization error and take it into account
  # when pruning.
  rem = np.copy(patches)
  count, error = 0, 0
  for region in get_regions_list(shape, 4*patchsize):
    cur, rem = split_patches(rem, region)
    for i in range(cur.shape[0]):
      x, y = cur[i,0], cur[i,1]
      if imp[y,x] - error > random.random():
        pruned[count,:] = [x, y]
        count += 1
        error += 1 - imp[y,x]
      else:
        error += 0 - imp[y,x]
  return pruned[:count,:]

def ImportanceSampling(data):
  buffers = [data[:,:,:3], data[:,:,19:22]] # color and normal
  metrics = ['relvar', 'variance']
  weights = [10.0, 1.0]
  imp = GetImportanceMap(buffers, metrics, weights, constants.patch_size)
  # get patches
  patches = SamplePatchesProg(data.shape[:2], (constants.patch_size, constants.patch_size), constants.n_patches)
  # prune(rejection sampling)
  pad = constants.patch_size // 2
  pruned = np.maximum(0, PrunePatches(data.shape[:2], patches + pad, constants.patch_size, imp) - pad)
  return (pruned + pad)

def Crop(data, gt, pos, patch_size):
  half_size = patch_size // 2
  px, py = pos
  return {
    "data":data[py-half_size:py+half_size,px-half_size:px+half_size,:],
    "gt":gt[py-half_size:py+half_size,px-half_size:px+half_size,:]
  }

def GetCroppedPatches(data, gt):
  patches = ImportanceSampling(data)
  cropped = [Crop(data, gt, pos, constants.patch_size) for pos in patches]
  return cropped

def CroppedFromFiles():
  cropped = []
  for f in glob.glob(envs.data_path + "*-16"):
    name = f[len(envs.data_path):f.index("-16")]
    print(f"load data and gt: {name}")
    gt = LoadHighSpp(envs.data_path + name + "-4096")
    print(f"size: {gt.shape[0]}x{gt.shape[1]}")
    data = LoadLowSpp(envs.data_path + name + "-16")
    newcropped = GetCroppedPatches(data, gt)
    cropped += newcropped
    print(f"get {len(newcropped)} patches, {len(cropped)} in total")
    
  return cropped

def TestDataFromFiles():
  test = []
  for f in glob.glob(envs.test_path + "*-16"):
    name = f[len(envs.test_path):f.index("-16")]
    print(f"load data and gt: {name}")
    gt = LoadHighSpp(envs.test_path + name + "-4096")
    print(f"size: {gt.shape[0]}x{gt.shape[1]}")
    data = LoadLowSpp(envs.test_path + name + "-16")
    test.append({"data":data, "gt":gt})
  return test

if __name__ == "__main__":
  bathroom_gt = LoadHighSpp(envs.data_path + "bathroom-4096")
  print(bathroom_gt.shape)
  bathroom_data, names_and_channels = LoadLowSpp(envs.data_path + "bathroom-16", True)
  print(names_and_channels)
