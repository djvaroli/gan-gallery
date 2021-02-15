from matplotlib import pyplot as plt
import numpy as np
import os

import tensorflow as tf


def generate_and_save_images(model_predcitions, save_location):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  model_predcitions = tf.cast(model_predcitions * 127.5 + 127.5, dtype=tf.int32) # convert to int for plotting

  fig, axs = plt.subplots(1, model_predcitions.shape[0], figsize=(16, 16))

  for i, ax in enumerate(axs.flatten()):
    plt.sca(ax)
    ax.imshow(model_predcitions[i])
    ax.axis('off')
  
  plt.savefig(save_location)
  plt.close()