
# DATA
CHARS_ALPHABETH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
EOS_CHAR = "$"
CHARS_LENGTH = 2
IMAGE_H, IMAGE_W = 32, 32

# TRAIN
CHECKPOINT_DIR = "./data/model"
LOG_DIR = "./data/logs"
BATCH_SIZE = 64
STEPS_PER_EPOCH = 32
NUM_EPOCHS = 5000

# OPTIMIZE GENERATOR ONLY
GEN_LR = .001

# ### OPTIMIZE WGAN-GP
# USE_WGAN_GP = True
# DIS_LR = .0005
# DIS_LR_DECAY_EPOCH = 250
# DIS_LR_DECAY = .5
# DIS_BETA_1 = .0
# DIS_BETA_2 = .9
# WGAN_GP_LAMBDA = 5.
# ### OPTIMIZE VANILLA GAN
USE_WGAN_GP = False
DIS_LR = .00005
DIS_LR_DECAY_EPOCH = 20000
DIS_LR_DECAY = .7
DIS_BETA_1 = .5
DIS_BETA_2 = .9

# MODEL
EMBED_SIZE = 8
RNN_SIZE = 16
LATENT_DIM = 0
NOISE_VAR = .05
DROP_PROB = .3
USE_MBD = False
MBD_KERNELS = 25
MBD_DIMS = 5
