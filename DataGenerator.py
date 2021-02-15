import os
import json
import random
from time import time
from PIL import Image
import logging
from logging import getLogger

import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import gensim.downloader 

logger = getLogger("DataGenerator")
logger.setLevel(logging.INFO)


class ImagesWithLabelsDataGenerator(Sequence):
    def __init__(
        self,
        index_to_label_fp = "gan_paintings_organized_data/index_to_labels_512.json", 
        label_to_indices_fp = "gan_paintings_organized_data/labels_to_indices_512.json", 
        image_dir_path = "gan_paintings_organized_data/indexed_paintings_512",
        image_shape = (512, 512, 3),
        batch_size = 20,
        preload_images = False,
        word_embedding_dim = 300
    ) -> None:
        # load in the forward and backward index - label mappings
        super(ImagesWithLabelsDataGenerator, self).__init__()
        print("Loading index <-> labels mappings")
        self.index_to_label = self.load_in_json_file(index_to_label_fp)
        self.label_to_indices = self.load_in_json_file(label_to_indices_fp)
        self.labels = list(self.label_to_indices.keys())
        
        # data for returning batches
        self.image_shape = image_shape
        self.image_dir_path = image_dir_path
        self.batch_size = batch_size
        self.shuffled_image_indices = self.load_shuffled_image_indices()

        # load glove vectors
        self.word_embedding_dim = word_embedding_dim
        print("Downloading GloVe embeddings, this may take a second.")
        self.glove_vectors = gensim.downloader.load(f"glove-wiki-gigaword-{word_embedding_dim}")
        print("Download completed")

    def __len__(self):
        return int(np.ceil(len(self.shuffled_image_indices) / self.batch_size))
    
    def __getitem__(self, index):
        # initialize arrays
        correct_labels_array = np.zeros((self.batch_size, self.word_embedding_dim)) # embedded correct labels
        wrong_labels_array = np.zeros((self.batch_size, self.word_embedding_dim)) # embedded wrong labels
        correct_images_array = np.zeros((self.batch_size, *self.image_shape))
        wrong_images_array = np.zeros((self.batch_size, *self.image_shape))

        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        image_indices = self.shuffled_image_indices[start_index: end_index]

        for i, image_index in enumerate(image_indices):
            correct_labels = self.index_to_label.get(image_index)
            correct_labels_split_list = correct_labels.split(",") # split into a list of labels
            embedded_correct_labels = self.embed_words(correct_labels_split_list)
            correct_labels_array[i] = embedded_correct_labels

            correct_image_array = self.load_images_as_np_array(image_index)
            correct_images_array[i] = correct_image_array

            wrong_labels, wrong_image_array = self.get_wrong_labels_and_images(correct_labels)
            wrong_labels_split_list = wrong_labels[0].split(",")
            embedded_wrong_labels = self.embed_words(wrong_labels_split_list)
            wrong_labels_array[i] = embedded_wrong_labels
            wrong_images_array[i] = wrong_image_array

        correct_labels_array = tf.convert_to_tensor(correct_labels_array)
        wrong_labels_array = tf.convert_to_tensor(wrong_labels_array)
        correct_images_array = tf.convert_to_tensor(correct_images_array)
        wrong_images_array = tf.convert_to_tensor(wrong_images_array)

        return (correct_labels_array, correct_images_array, wrong_labels_array, wrong_images_array)
    
    def load_in_json_file(self, local_path_to_file):
        with open(local_path_to_file, "r+") as f:
            json_data = json.load(f)
        return json_data
    
    def load_images_as_np_array(self, image_indices:list):
        if not isinstance(image_indices, list):
            image_indices = [image_indices]
        shape = (len(image_indices), *self.image_shape)
        image_arrays = np.zeros(shape)
        
        for i, idx in enumerate(image_indices):
            image_arrays[i] = self._load_image_as_np_array(idx) # will standerdize by default
            
        return tf.convert_to_tensor(image_arrays)
    
    def embed_words(self, words):
        combined_word_vector = np.zeros([1, self.word_embedding_dim])
        for word in words:
            if word == "diningtable":
                word = "table"
            combined_word_vector += self.glove_vectors.word_vec(word)[np.newaxis, :]
        
        return combined_word_vector
    
    def get_wrong_labels_and_images(self, correct_labels):
        """
        return a pair of wrong_label, wrong_image given a list of correct labels
        """
        if not isinstance(correct_labels, list):
            correct_labels = [correct_labels]

        shape = (len(correct_labels), *self.image_shape)
        wrong_image_arrays = np.zeros(shape)
        wrong_labels = []
        selected_wrong_indices = []
        
        for i,label in enumerate(correct_labels):
            all_labels = self.labels.copy()
            all_labels.remove(label)
            wrong_label = random.choice(all_labels)
            wrong_labels.append(wrong_label)
            
            wrong_image_indices = self.label_to_indices[wrong_label]
            wrong_image_index = random.choice(wrong_image_indices)
            wrong_image_arrays[i] = self._load_image_as_np_array(wrong_image_index)
            selected_wrong_indices.append(wrong_image_index)
            
        return wrong_labels, tf.convert_to_tensor(wrong_image_arrays)
    
    def _load_image_as_np_array(self, image_index, normalize=True):
        path_to_image = f"{self.image_dir_path}/image_{image_index}"
        img = Image.open(path_to_image)
        image_as_array = np.asarray(img)
        img.close()
        if normalize:
            image_as_array = (image_as_array - 122.5) / 122.5
        
        return image_as_array
    
    def load_shuffled_image_indices(self):
        indices = list(self.index_to_label.keys())
        random.shuffle(indices)
        return indices

    def on_epoch_end(self):
        random.shuffle(self.shuffled_image_indices)


def get_mnist_dataset(
    buffer_size: int = 60000,
    batch_size: int = 256,
    normalize_data: bool = True
):
    (train_images, train_labels), (_,_) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    if normalize_data:
        train_images = (train_images - 127.5) / 127.5 
    
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)




