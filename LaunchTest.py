__author__ = 'jasper.zuallaert'

# This file is called by either our bash script, or manually, to initiate training of a specified model.
import TestLauncher
import sys
import warnings
import tensorflow as tf

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Configure GPU memory growth (TF2 equivalent of allow_growth)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"GPU config error: {e}")

TestLauncher.runTest(sys.argv[1])
