# This file handles all the model training
from argparse import ArgumentParser
from time import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from generator import GeneratorModelMNIST, generator_loss
from discriminator import DiscriminatorModelMNIST, discriminator_loss
from DataGenerator import get_mnist_dataset
import ml_utils
import general_utils


def train_vanilla_gan_on_mnist(args):
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name

    generator_model = GeneratorModelMNIST(**args.__dict__)
    discriminator_model = DiscriminatorModelMNIST(**args.__dict__)
    generator_optimizer = Adam(1e-4)
    discriminator_optimizer = Adam(1e-4)

    data_generator = get_mnist_dataset(batch_size=batch_size)

    noise_dim = args.noise_dim
    num_examples_to_generate = args.num_examples_to_generate
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    plotting_callback = ml_utils.PlotAndSaveImages(
        test_input=seed,
        model=generator_model,
        model_name=model_name
    )

    ckpt = ml_utils.SimpleGANCheckPoint(gen_model=generator_model, disc_model=discriminator_model, model_name=model_name)
    plotting_callback.on_epoch_end(epoch=-1)
    
    for epoch in range(n_epochs):
        start = time()
        for i, image_batch in enumerate(data_generator):
            input_noise = tf.random.normal(shape=(image_batch.shape[0], noise_dim))
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator_model(input_noise, training=True)

                true_output = discriminator_model(image_batch, training=True)
                fake_output = discriminator_model(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(true_output, fake_output)
                
                gen_gradients = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
                disc_gradients = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
                
                generator_optimizer.apply_gradients(zip(gen_gradients, generator_model.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator_model.trainable_variables))

                general_utils.smart_print(start, len(data_generator), i + 1, epoch + 1, n_epochs, gen_loss, disc_loss)

        logs = {"generator_loss": gen_loss, "discriminator_loss": disc_loss}
        ckpt.on_epoch_end(epoch=epoch, logs=logs)    
        plotting_callback.on_epoch_end(epoch=epoch)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gan-mnist")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--noise_dim", type=int, default=100, help="dimension of the noise to be used as input to the generator.")
    parser.add_argument("--image_size", type=int, default=28, help="The length/width of the image. Images are assumed to be square.")
    parser.add_argument("--num_examples_to_generate", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    
    args = parser.parse_args()
    train_vanilla_gan_on_mnist(args)