import tensorflow as tf
from tensorflow.keras import layers as layers
from tensorflow.initializers import GlorotNormal

def DiscriminatorModel(
    image_shape: tuple, 
    embedded_labels_shape: tuple
) -> tf.keras.Model:
  # takes an image and embedded labels as an input
  image_input = layers.Input(shape=image_shape)
  labels_input = layers.Input(shape=embedded_labels_shape)

  embedded_labels = layers.Dense(512)(labels_input)

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


def discriminator_loss(
    real_image_real_caption_output,
    real_image_wrong_caption_output,
    fake_image_real_caption_output,
    wrong_image_real_caption_output
):
    """
    real_image + real_caption => 1.0
    real_image + wrong_caption => 0.0 , wrong caption is selected at random from all possible captions
    fake_image + real_caption => 0.0
    wrong_image + real_caption => 0.0
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