

# DATA
CHARS_ALPHABETH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EOS_CHAR = "$"
CHARS_LENGTH = 2
IMAGE_H, IMAGE_W = 32, 32

# TRAIN
CHECKPOINT_DIR = "./data/model"
LOG_DIR = "./data/logs"
BATCH_SIZE = 64
STEPS_PER_EPOCH = 32
NUM_EPOCHS = 5000

# OPTIMIZE
GEN_LR = .001
DIS_LR = .0001
DIS_LR_HALF_EPOCH = 200
DIS_LR_DECAY = .7
DIS_BETA_1 = .0
DIS_BETA_2 = .9
WGAN_GP_LAMBDA = 5.

# MODEL
EMBED_SIZE = 8
RNN_SIZE = 16
NOISE_DIM = 0
DROP_PROB = .3
USE_WGAN_GP = True






