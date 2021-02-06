import os
import json
import random
from time import time
from PIL import Image
from logging import getLogger

import numpy as np
import tensorflow as tf
import gensim.downloader 

logger = getLogger("DataGenerator")

logger.info("Downloading GloVe embeddings.")
glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")
logger.info("Download completed")

class DataGenerator():
    def __init__(
        self,
        index_to_label_fp = "gan_paintings_organized_data/index_to_labels_512.json", 
        label_to_indices_fp = "gan_paintings_organized_data/labels_to_indices_512.json", 
        image_dir_path = "gan_paintings_organized_data/indexed_paintings_512",
        image_shape = (512, 512, 3),
        batch_size = 20,
        preload_images = False
    ) -> None:
        # load in the forward and backward index - label mappings
        self.index_to_label = self.load_in_json_file(index_to_label_fp)
        self.label_to_indices = self.load_in_json_file(label_to_indices_fp)
        self.labels = list(self.label_to_indices.keys())
        
        # data for returning batches
        self.image_shape = image_shape
        self.image_dir_path = image_dir_path
        self.batch_size = batch_size
        self.shuffled_image_indices = self.load_shuffled_image_indices()
        
    def __len__(self):
        return int(np.ceil(len(self.shuffled_image_indices) / self.batch_size))
    
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size

        indices = self.shuffled_image_indices[start_index: end_index]
        batch_correct_labels = [self.index_to_label.get(i) for i in indices] # get the text labels
        batch_correct_labels_split = [labels.split(",") for labels in batch_correct_labels] # split the labels into separate words
        embedded_batch_correct_labels = tf.convert_to_tensor([self.embed_words(labels) for labels in batch_correct_labels_split])
        word_vec_dim = embedded_batch_correct_labels.shape[-1]
        embedded_batch_correct_labels = tf.reshape(embedded_batch_correct_labels, (self.batch_size, word_vec_dim))
        correct_image_arrays = self.load_images_as_np_array(indices)
        
        batch_wrong_labels, wrong_image_arrays = self.get_wrong_labels_and_images(batch_correct_labels)
        batch_wrong_labels_split = [labels.split(",") for labels in batch_wrong_labels]
        embedded_batch_wrong_labels = tf.convert_to_tensor([self.embed_words(labels) for labels in batch_wrong_labels_split])
        embedded_batch_wrong_labels = tf.reshape(embedded_batch_wrong_labels, (self.batch_size, word_vec_dim))

        return (embedded_batch_correct_labels, correct_image_arrays, embedded_batch_wrong_labels, wrong_image_arrays)
    
    def load_in_json_file(self, local_path_to_file):
        with open(local_path_to_file, "r+") as f:
            json_data = json.load(f)
        return json_data
    
    def load_images_as_np_array(self, image_indices):
        shape = (len(image_indices), *self.image_shape)
        image_arrays = np.zeros(shape)
        
        for i, idx in enumerate(image_indices):
            image_arrays[i] = self._load_image_as_np_array(idx) # will standerdize by default
            
        return tf.convert_to_tensor(image_arrays)
    
    def embed_words(self, words):
        combined_word_vector = np.zeros([1, 300])
        for word in words:
            if word == "diningtable":
                word = "table"
            combined_word_vector += glove_vectors.word_vec(word)[np.newaxis, :]
        
        return combined_word_vector
    
    def get_wrong_labels_and_images(self, correct_labels):
        """
        return a pair of wrong_label, wrong_image given a list of correct labels
        """
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
    
    def _load_image_as_np_array(self, image_index, standerdize=True):
        path_to_image = f"{self.image_dir_path}/image_{image_index}"
        img = Image.open(path_to_image)
        image_as_array = np.asarray(img)
        img.close()
        if standerdize:
            image_as_array = image_as_array / 255.
        
        return image_as_array
    
    def load_shuffled_image_indices(self):
        indices = list(self.index_to_label.keys())
        random.shuffle(indices)
        return indices

