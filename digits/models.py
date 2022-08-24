import tensorflow as tf
import numpy as np

from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.optimizers import Adam

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

# inherit from sequential?

class Generator(tf.keras.Sequential):

  def __init__(self):
    super(Generator, self).__init__()

    self.add(Dense(units=256, input_dim=100))
    self.add(LeakyReLU(0.2))
    self.add(Dense(units=512))
    self.add(LeakyReLU(0.2))
    self.add(Dense(units=1024))
    self.add(LeakyReLU(0.2))
    self.add(Dense(units=784, activation='tanh'))

    self.compile(loss = 'binary_crossentropy',
                      optimizer = Adam(learning_rate=0.0002, beta_1=0.5))


class Discriminator(tf.keras.Sequential):

  def __init__(self):
    super(Discriminator, self).__init__()


    self.add(Dense(units = 1024, input_dim = 784))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))
    self.add(Dense(units = 512))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))

    self.add(Dense(units=256))
    self.add(LeakyReLU(0.2))

    self.add(Dense(units=1, activation='sigmoid'))

    self.compile(loss = 'binary_crossentropy',
                          optimizer = Adam(learning_rate=0.0002, beta_1=0.5))


class GAN(tf.keras.Model):

  def __init__(self, generator, discriminator):
      # Initialize random noise with generator
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    super(GAN, self).__init__(inputs = gan_input, outputs = gan_output)
    self.compile(loss = 'binary_crossentropy',
                 optimizer = 'adam')

    self.generator = generator
    self.discriminator = discriminator

  def write_log(self, callback, names, logs, batch_no):
      for name, value in zip(names, logs):
          summary = tf.Summary()
          summary_value = summary.value.add()
          summary_value.simple_value = value
          summary_value.tag = name
          callback.writer.add_summary(summary, batch_no)
          callback.writer.flush()


  def train_gan_on_batch(self, image_batch):

    batch_size = len(image_batch)

    noise = np.random.normal(0,1, [batch_size, 100])

    # Generate fake MNIST images from noised input
    generated_images = self.generator.predict(noise)

    # Get a random set of  real images

    #Construct different batches of real and fake data
    X = np.concatenate([image_batch, generated_images])

    # Labels for generated and real data
    y_dis = np.zeros(2 * batch_size)
    y_dis[:batch_size] = 0.9

    #Pre train discriminator on  fake and real data  before starting the gan.
    self.discriminator.trainable=True
    self.generator.trainable=False

    self.discriminator.train_on_batch(X, y_dis)

    #Tricking the noised input of the Generator as real data
    noise = np.random.normal(0,1, [batch_size, 100])
    y_gen = np.ones(batch_size)

    # During the training of gan,
    # the weights of discriminator should be fixed.
    #We can enforce that by setting the trainable flag
    self.discriminator.trainable=False
    self.generator.trainable=True

    #training  the GAN by alternating the training of the Discriminator
    #and training the chained GAN model with Discriminator’s weights freezed.
    return self.train_on_batch(noise, y_gen)
