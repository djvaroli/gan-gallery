from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers as layers
from tensorflow.keras.initializers import GlorotNormal



def DiscriminatorModelBasic(
    input_shape: tuple, 
    *args,
    **kwargs
):
  # takes an image as input
  image_input = layers.Input(shape=input_shape)

  # perform convolution on the image
  x = layers.Conv2D(64, (5,5), strides=(2,2), padding="same")(image_input)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128, (5,5), strides=(2,2), padding="same")(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Flatten()(x) 
  x = layers.Dense(1, activation="sigmoid", kernel_regularizer=GlorotNormal)(x) 

  model = tf.keras.Model(image_input, x)

  return model


def DiscriminatorModelMNIST(*args, **kwargs):
    input = layers.Input(shape=(28, 28, 1))

    x = convultion_block(input, filters=64)
    x = convultion_block(x, filters=128)
    x = convultion_block(x, filters=256)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model(input, x)


def DiscriminatorModelWithEmbeddedText(
    input_shape: tuple, 
    embedded_labels_shape: tuple
) -> tf.keras.Model:
  # takes an image and embedded labels as an input
  image_input = layers.Input(shape=input_shape)
  labels_input = layers.Input(shape=embedded_labels_shape)

  embedded_labels = layers.Dense(128)(labels_input)

  # perform convolution on the image
  x = layers.Conv2D(64, (5,5), strides=(2,2), padding="same")(image_input)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128, (5,5), strides=(2,2), padding="same")(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Flatten()(x) 
  x = layers.Concatenate(axis=1)([x, embedded_labels]) # concatenate along row
  x = layers.Dense(1, activation="sigmoid", kernel_regularizer=GlorotNormal)(x) 

  model = tf.keras.Model([image_input,labels_input], x)

  return model


def convultion_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (5, 5),
    strides: Tuple[int, int] = (2, 2),
    padding: str = "same",
    dropout_prob: float = 0.3
):
  x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
  x = layers.LeakyReLU()(x)

  return layers.Dropout(dropout_prob)(x)


def discriminator_loss_with_embedded_text(
    real_image_real_caption_output,
    real_image_wrong_caption_output,
    fake_image_real_caption_output,
    wrong_image_real_caption_output
):
    """
    real_image + real_caption => 1.0
    real_image + wrong_caption => 0.0
    fake_image + real_caption => 0.0
    Final loss is 1 * log(D(real_image + real_caption)) + 0.5 * (log(D(real_image + wrong_caption)) + log(fake_image + real_caption))
    as per Algorithm 1 in https://arxiv.org/pdf/1605.05396.pdf
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    loss_0 = cross_entropy(
        tf.random.normal(real_image_real_caption_output.shape, mean=1.0, stddev=0.05),
        real_image_real_caption_output
    )

    loss_1 = cross_entropy(
        tf.random.normal(real_image_wrong_caption_output.shape, mean=0.0, stddev=0.05),
        real_image_wrong_caption_output
    )
        
    loss_2 = cross_entropy(
        tf.random.normal(fake_image_real_caption_output.shape, mean=0.0, stddev=0.05),
        fake_image_real_caption_output
    )
    
    loss_3 = cross_entropy(
        tf.random.normal(wrong_image_real_caption_output.shape, mean=0.0, stddev=0.05),
        wrong_image_real_caption_output
    )
    
    return .25*loss_0 + 0.25*loss_1 + 0.25*loss_2 + 0.25*loss_3


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss
