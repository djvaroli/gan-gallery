from matplotlib import pyplot as plt
import numpy as np
import os

import tensorflow as tf

def generate_and_save_images(model, epoch, test_input, dir_prefix, show_images=False):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  predictions = tf.cast(predictions * 122.5 + 122.5, dtype=tf.int32) # convert to int for plotting

  fig, axs = plt.subplots(1, predictions.shape[0], figsize=(16, 16))

  for i, ax in enumerate(axs.flatten()):
    ax.imshow(predictions[i])
    ax.axis('off')

  if os.path.isdir(dir_prefix) is False:
    os.mkdir(dir_prefix)
  
  plt.savefig(f'{dir_prefix}/image_at_epoch_{epoch: 04d}.png')

  if show_images:
    plt.show()