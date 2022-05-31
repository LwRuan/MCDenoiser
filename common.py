from calendar import EPOCH


class envs:
  data_path = "../pngs/"
  test_path = "../test-pngs/"

class switches:
  albedo_div = True
  mode = "KPCN" # "KPCN" or "DPCN"

class constants:
  patch_size = 64
  n_patches = 400
  # precent_validation = 0.1
  output_kernel_size = 17
  eps = 0.001

class nnconfigs:
  input_channels = 26 # don't change
  L = 9 # number of convolutional layers
  n_kernels = 100 # number of kernels in each layer
  kernel_size = 5 # size of kernel (square)
  hidden_channels = 100

class trainconfigs:
  learning_rate = 1e-5
  epochs = 2000