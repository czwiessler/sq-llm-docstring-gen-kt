SIMPLE_RUN = False
END_TO_END = True


# Text encoder conf
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
ALPHA_SIZE = len(ALPHABET)
CHAR_DEPTH = 201
BATCH_SIZE = 40
ENCODER_TRAINING_PATH = 'assets/encoder_train'
PRE_PROCESSING_THREADS = 20
NUM_GPU_TXT_ENCODER = 2
FORCE_GPU_TEXT_ENCODER = False

# GAN Config
NUM_D_FILTER = 64
NUM_G_FILTER = 128
ENCODED_TEXT_SIZE = 128
GAN_BATCH_SIZE = 64
NUM_GPU = 4
FORCE_GPU = False
GAN_TOWER_BATCH_SIZE = int(GAN_BATCH_SIZE/NUM_GPU) # Batch_Size_Per_Tower
ENABLE_RESIDUAL_NET = False