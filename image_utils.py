from matplotlib import pyplot as plt
import numpy as np
import os

def generate_and_save_images(model, epoch, test_input, dir_prefix, plot_images=False):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  if plot_images:
    fig = plt.figure(figsize=(16,16))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')

  if os.path.isdir(dir_prefix) is False:
    os.mkdir(dir_prefix)
  
  plt.savefig(f'{dir_prefix}/image_at_epoch_{epoch: 04d}.png')

  if plot_images:
    plt.show()