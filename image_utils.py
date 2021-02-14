from matplotlib import pyplot as plt
import numpy as np
import os

import tensorflow as tf


def generate_and_save_images(test_input, save_location):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = tf.cast(predictions * 127.5 + 127.5, dtype=tf.int32) # convert to int for plotting

  fig, axs = plt.subplots(1, predictions.shape[0], figsize=(16, 16))

  for i, ax in enumerate(axs.flatten()):
    plt.sca(ax)
    ax.imshow(predictions[i])
    ax.axis('off')
  
  plt.savefig(save_location)
  plt.close()