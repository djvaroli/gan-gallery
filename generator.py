import tensorflow as tf
from tensorflow.keras import layers as layers


def GeneratorModel(
    input_dim, 
    num_channels
) -> tf.keras.Model:
    input = layers.Input(shape=(input_dim, ))
    x = layers.Dense(8*8*256, use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # reshape
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding="same", use_bias=False, activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(16, (5,5), strides=(2,2), padding="same", use_bias=False, activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(num_channels, (5,5), strides=(2,2), padding="same", use_bias=False, activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    model = tf.keras.Model(input, x)
    return model


def generator_loss(
    fake_output: tf.Tensor
):
    """
    compare the descriminator output for the fake input to 1.0
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = cross_entropy(tf.random.normal(fake_output.shape, mean=1.0, stddev=0.05), fake_output)

    return loss