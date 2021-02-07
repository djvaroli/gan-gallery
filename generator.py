import tensorflow as tf
from tensorflow.keras import layers as layers


def GeneratorModel(
    input_dim, 
    num_channels,
    batch_norm_momentum=0.9,
    alpha=0.2,
    start_deconv_dim=512,
    start_dense_dim=2
) -> tf.keras.Model:
    
    input = layers.Input(shape=(input_dim, ))
    x = layers.Dense(start_dense_dim*start_dense_dim*1024, use_bias=False)(input)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Reshape((start_dense_dim, start_dense_dim, 1024))(x)
    
    num_deconv_layers = np.log2(512 // start_dense_dim * 1.).astype(np.int32) - 1

    dim = start_deconv_dim
    for i in range(1, num_deconv_layers + 1):
        conv_2d_transpose_layer = layers.Conv2DTranspose(
            dim, (5,5), strides=(2,2), padding='same', use_bias=False, kernel_initializer=GlorotNormal, activation="tanh"
        )
        x = conv_2d_transpose_layer(x)
        x = layers.BatchNormalization(momentum=batch_norm_momentum)(x)
        x = layers.LeakyReLU(alpha=alpha)(x)
        dim = dim // 2
        
    x = layers.Conv2DTranspose(num_channels, (5,5), strides=(2,2), padding="same", use_bias=False, activation="tanh", kernel_initializer=GlorotNormal)(x)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)

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