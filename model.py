from typing import *
from argparse import ArgumentParser
from time import time
from datetime import datetime as dt
import os

import tensorflow as tf

from generator import GeneratorModel, generator_loss
from discriminator import DiscriminatorModel, discriminator_loss
from image_utils import generate_and_save_images
from general_utils import smart_print
from DataGenerator import DataGenerator


def train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    real_images,
    real_captions,
    wrong_images,
    wrong_captions,
    args
):
    """
    generate some noise -> pass into generator to generate fake images
    get the real caption for the imaage, get a wrong caption randomly sampled from all other captions
    obtain a wrong image from a different class

    pass fake images and real images and wrong images through discriminator with the captions
    """

    batch_size = args.batch_size
    noise_dim = args.gen_input_dim
    
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_image_real_caption_output = discriminator([real_images, real_captions], training=True)
        real_image_fake_wrong_output = discriminator([real_images, wrong_captions], training=True)
        fake_image_real_caption_output = discriminator([generated_images, real_captions], training=True)
        wrong_image_real_caption_output = discriminator([wrong_images, real_captions], training=True)

        gen_loss = generator_loss(fake_image_real_caption_output)
        disc_loss = discriminator_loss(
            real_image_real_caption_output, real_image_fake_wrong_output, fake_image_real_caption_output, wrong_image_real_caption_output
            )

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(
    dataset, 
    generator, 
    discriminator,
    generator_oprimizer,
    discriminator_optimizer,
    checkpoint,
    checkpoint_prefix,
    args
):  
    epochs = args.n_epochs
    save_every_n_epochs = args.save_every_n_epochs
    gen_input_dim = args.gen_input_dim # noise_dim + 300 for the gensim word embeddings
    test_noise = tf.random.normal([4, gen_input_dim])

    for i in range(epochs):
        start = time()

        for j, (true_labels, real_images, wrong_labels, wrong_images) in enumerate(dataset):
            gen_loss, disc_loss = train_step(
                generator, 
                discriminator, 
                generator_oprimizer, 
                discriminator_optimizer, 
                real_images=real_images, 
                real_captions=true_labels,
                wrong_images=wrong_images, 
                wrong_captions=wrong_labels, 
                args=args
            )
            smart_print(start, len(dataset), j + 1, i + 1, epochs, gen_loss, disc_loss)

        if (i + 1) % save_every_n_epochs == 0:
            print(f"Saving weights for epoch {i + 1}")
            checkpoint.save(file_prefix = checkpoint_prefix)

        generate_and_save_images(generator, i, test_noise, True)


def main(
    args
):
    # some hyperparameters
    batch_size = args.batch_size
    gen_input_dim = args.gen_input_dim
    embedded_labels_dim = args.embedded_labels_dim
    num_channels = args.num_channels
    image_dim = args.image_dim

    # shapes of inputs to the discriminator
    disc_input_image_shape = (image_dim, image_dim, num_channels)
    embedded_labels_shape = (embedded_labels_dim, )

    # this will be the dataset, which we will iterate over during training
    data_generator = DataGenerator(batch_size=batch_size)

    # the models themselves and the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator = GeneratorModel(input_dim=gen_input_dim, num_channels=num_channels)
    discriminator = DiscriminatorModel(image_shape=disc_input_image_shape, embedded_labels_shape=embedded_labels_shape)

    # checkpointing to save intermediate model states
    model_date_key = dt.now().isoformat()
    checkpoint_dir = f"./gan_paintings_w_labels_{model_date_key}"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    # train the model
    train(data_generator, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--save_every_n_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--gen_input_dim', help="the dimension of the input vector to the generator.", type=int)
    parser.add_argument('--embedded_labels_dim', help="Dimension of the word-embeddings of labels.", type=int, default=300)
    parser.add_argument('--image_dim', help="Length of side of image in pixels. Images are assumed to be square", type=int, default=512)
    parser.add_argument('--num_channels', help="Number of channels of the images in the dataser", type=int, default=3)

    args = parser.parse_args()

    main(args)




