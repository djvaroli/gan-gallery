from pathlib import Path
import os
from typing import *
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import image_utils

class SimpleModelCheckPoint(ModelCheckpoint):
    """
    Wrapper around the tf.keras.callbacks.ModelCheckpoint with the added
    feature of a default file path
    """
    def __init__(
        self, 
        filepath=None, 
        model_name="saved_model", 
        model = None,
        *args, 
        **kwargs
    ):
        if filepath is None:
            date_key = dt.today().strftime("%Y-%m-%dT%H-%M")
            dir = Path(f"{model_name}_{date_key}")
            if os.path.isdir(dir) is False:
                os.mkdir(dir)
            filepath = str(dir / "weights.{epoch:02d}-{loss:.2f}.hdf5")
        
        super(SimpleModelCheckPoint, self).__init__(filepath, *args, **kwargs)


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
