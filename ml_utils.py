from pathlib import Path
import os
from typing import *
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import image_utils

class SimpleGANCheckPoint(Callback):
    """
    Wrapper around the tf.keras.callbacks.ModelCheckpoint with the added
    feature of a default file path
    """
    def __init__(
        self, 
        gen_model: tf.keras.models.Model,
        disc_model: tf.keras.models.Model,
        model_name: str = "saved_model", 
        filepath: str = None,
        save_frequency: int = 1,
        *args, 
        **kwargs
    ):
        self.save_frequency = save_frequency
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.model_name = model_name

        if not filepath:
            self.save_directory = self.__generate_save_directory_name()
            self.filepath = self.__generate_save_path_template()
        else:
            self.filepath = filepath

    def __generate_save_directory_name(self):
        date_key = dt.today().strftime("%Y-%m-%dT%H-%M")
        directory = Path(f"{self.model_name}_{date_key}_saved_weights")
        if os.path.isdir(directory) is False:
            os.mkdir(directory)

        return directory 
    
    def __generate_save_path_template(self):
        return str(self.save_directory / "weights-{model_name}-{epoch:02d}-{loss:.2f}.hdf5")

    def on_epoch_end(self, epoch, logs=None):
        generator_loss = logs['generator_loss']
        discriminator_loss = logs['discriminator_loss']

        if epoch % self.save_frequency == 0:
            gen_weights_path = self.filepath.format(model_name="generator", epoch=epoch, loss=generator_loss)
            disc_weights_path = self.filepath.format(model_name="discriminator", epoch=epoch, loss=discriminator_loss)

            self.gen_model.save_weights(gen_weights_path)
            self.disc_model.save_weights(disc_weights_path)


class PlotAndSaveImages(Callback):
    def __init__(
        self,
        test_input: tf.Tensor,
        save_directory: Union[str, Path] = None,
        model_name: str = "saved_model",
        save_frequency: str = "epoch",
        patience: int = 1,
        model: tf.keras.models.Model = None,
        *args,
        **kwargs
    ) -> None:
        self.test_input = test_input
        self.model_name = model_name
        self.save_frequency = save_frequency
        self.patience = patience
        self.save_directory = save_directory if save_directory else self.__generate_save_directory_name()
        self.save_path = self.__generate_save_path_template()

        if model:
            self.model = model
        super(PlotAndSaveImages, self).__init__()
    
    def on_epoch_end(self, epoch, keys=None):
        if self.save_frequency == "epoch" and epoch % self.patience == 0:
            test_prediction = self.model.predict(self.test_input)
            savepath = self.save_path.format(epoch=epoch)
            image_utils.generate_and_save_images(test_prediction, savepath)
        
    def __generate_save_directory_name(self):
        model_name = self.model_name
        date_key = dt.today().strftime("%Y-%m-%dT%H-%M")
        dir = Path(f"{model_name}_{date_key}_output_images")
        if os.path.isdir(dir) is False:
            os.mkdir(dir)
        return dir 
    
    def __generate_save_path_template(self):
        return str(self.save_directory / "image_at_epoch_{epoch:02d}.png")
