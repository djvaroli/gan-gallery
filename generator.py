from typing import *

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers  
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras import activations
import numpy as np


def GeneratorModelBasic(
    noise_dim: int, 
    num_channels: int,
    batch_norm_momentum=0.9,
    alpha=0.2,
    start_deconv_filters=512,
    start_dense_dim=2,
    *args,
    **kwargs
) -> tf.keras.Model:
    
    input = layers.Input(shape=(noise_dim, ))
    
    x = layers.Dense(start_dense_dim*start_dense_dim*1024, use_bias=False)(input)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Reshape((start_dense_dim, start_dense_dim, 1024))(x)
    
    num_deconv_layers = np.log2(512 // start_dense_dim * 1.).astype(np.int32) - 1
    filters = start_deconv_filters
    for i in range(1, num_deconv_layers + 1):
        x = conv_2d_transpose_layer_block(x, filters)
        filters = filters // 2

    x = conv_2d_transpose_layer_block(x, num_channels, activation="tanh")

    model = tf.keras.Model(input, x)
    return model


def GeneratorModelMNIST(
    noise_dim: int,
    batch_norm_momentum=0.9,
    alpha=0.2,
    *args,
    **kwargs
) -> tf.keras.Model:
    input = layers.Input(shape=(noise_dim,))

    x = layers.Dense(7*7*256, use_bias=False)(input)
    x = layers.Dropout(0.5)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Reshape((7, 7, 256))(x)

    x = conv_2d_transpose_layer_block(x, 128, strides=(1,1)) # (None, 7, 7, 128)
    x = conv_2d_transpose_layer_block(x, 64) # (None, 14, 14, 64)
    x = conv_2d_transpose_layer_block(x, 32, strides=(1,1)) # (None, 14, 14, 32)
    x = conv_2d_transpose_layer_block(x, 1, activation="tanh") # (None, 28, 28, 1)

    return tf.keras.Model(input, x)


def GeneratorModelWithEmbeddedText(
    input_dim, 
    num_channels,
    batch_norm_momentum=0.9,
    alpha=0.2,
    start_deconv_filters=512,
    start_dense_dim=2,
    embedded_labels_dim=300
) -> tf.keras.Model:
    
    input = layers.Input(shape=(input_dim, ))
    embedded_labels_input = layers.Input(shape=(embedded_labels_dim, ))

    embedded_labels = layers.Dense(128)(embedded_labels_input)
    embedded_labels = layers.LeakyReLU(alpha=alpha)(embedded_labels)

    input = layers.Concatenate(axis=1)([input, embedded_labels])

    x = layers.Dense(start_dense_dim*start_dense_dim*1024, use_bias=False)(input)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Reshape((start_dense_dim, start_dense_dim, 1024))(x)
    
    num_deconv_layers = np.log2(512 // start_dense_dim * 1.).astype(np.int32) - 1
    filters = start_deconv_filters
    for i in range(1, num_deconv_layers + 1):
        x = conv_2d_transpose_layer_block(x, filters)
        filters = filters // 2

    x = conv_2d_transpose_layer_block(x, num_channels, activation="tanh")

    model = tf.keras.Model(input, x)
    return model


def conv_2d_transpose_layer_block(
    x: tf.Tensor, 
    filters: int, 
    kernel_size: Tuple[int, int] = (5, 5),
    strides: Tuple[int, int] = (2, 2),
    padding: str = "same",
    use_bias: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = GlorotNormal,
    activation: Union[str, None] = None,
    batch_norm_momentum: float = 0.9,
    alpha: float = 0.2
):
    conv_2d_transpose_layer = layers.Conv2DTranspose(
        filters, kernel_size, strides, padding, use_bias=use_bias, kernel_initializer=kernel_initializer, activation=activation
    )
    x = conv_2d_transpose_layer(x)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)

    return x


def generator_loss(
    discriminator_output: tf.Tensor,
):
    """
    compare the descriminator output for the fake input to 1.0
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(discriminator_output), discriminator_output)