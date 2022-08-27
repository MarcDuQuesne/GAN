import tensorflow as tf
import numpy as np
from tqdm import tqdm
import keras

from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.optimizers import Adam

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model


class Generator(keras.Sequential):

  def __init__(self, input_dim, output_dim, units=32):
    super(Generator, self).__init__()

    self.input_dim = input_dim

    self.add(Dense(units=units, input_dim=input_dim))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))
    self.add(Dense(units=units*2))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))
    self.add(Dense(units=units*4))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))
    self.add(Dense(units=output_dim, activation='tanh'))

    self.compile(loss = 'binary_crossentropy',
                 optimizer = Adam(learning_rate=0.0002, beta_1=0.5))

class Discriminator(keras.Sequential):

  def __init__(self, input_dim, units=32):
    super(Discriminator, self).__init__()

    self.add(Dense(units = units*4, input_dim=input_dim))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))
    self.add(Dense(units = units*2))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))
    self.add(Dense(units=units))
    self.add(LeakyReLU(0.2))
    self.add(Dropout(0.3))

    self.add(Dense(units=1, activation='sigmoid'))

    self.compile(loss = 'binary_crossentropy',
                 optimizer = Adam(learning_rate=0.0002, beta_1=0.5))


class GAN():

  def __init__(self, generator, discriminator):

    self.generator = generator
    self.discriminator = discriminator

    # Initialize random noise with generator
    gan_input = Input(shape=(self.generator.input_dim,))
    x = self.generator(gan_input)
    gan_output = self.discriminator(x)

    # super(GAN, self).__init__(inputs = gan_input, outputs = gan_output)
    self.model = keras.Model(inputs = gan_input, outputs = gan_output)
    self.model.compile(loss = 'binary_crossentropy',
                       optimizer = Adam(learning_rate=0.0002, beta_1=0.5))

  def write_log(self, callback, names, logs, batch_no):
      for name, value in zip(names, logs):
          summary = tf.Summary()
          summary_value = summary.value.add()
          summary_value.simple_value = value
          summary_value.tag = name
          callback.writer.add_summary(summary, batch_no)
          callback.writer.flush()


  def fit(self, X, batch_size, epochs=10, tensorboard=None):
    """
    training  the GAN by alternating the training of the Discriminator
    and training the chained GAN model with Discriminator’s weights freezed.
    """

    for epoch_id in tqdm(range(1,epochs+1)):
        for batch_id in range(batch_size):
            # Get a random set of real images
            image_batch = X[np.random.randint(low=0, high=X.shape[0], size=batch_size)]
            logs = self.train_on_single_batch(image_batch)
        if tensorboard:
          tensorboard.on_epoch_end(epoch_id, logs)

  def train_on_single_batch(self, image_batch):

    batch_size = len(image_batch)
    noise = np.random.normal(0, 1, [batch_size, self.generator.input_dim])

    # Generate fake MNIST images from noised input
    generated_images = self.generator.predict(noise)
    # Construct different batches of real and fake data
    X = np.concatenate([image_batch, generated_images])

    # Labels for generated and real data
    # 1 - real image
    # 0 - generated
    y_dis = np.zeros(2 * batch_size)
    y_dis[:batch_size] = 0.9

    # Pre train discriminator on  fake and real data  before starting the gan.
    self.discriminator.trainable = True
    self.generator.trainable = False

    # discriminator_loss = self.discriminator.train_on_batch(X, y_dis)
    # discriminator_loss = self.discriminator.fit(X, y_dis, batch_size=batch_size, epochs=1, verbose=0, callbacks=[])
    noise2 = np.random.normal(0, 1, [batch_size*2, self.generator.input_dim])
    discriminator_loss = self.model.fit(noise2, y_dis, batch_size=batch_size, epochs=1, verbose=0, callbacks=[])

    # Tricking the noised input of the Generator as real data
    noise = np.random.normal(0, 1, [batch_size, self.generator.input_dim])
    y_gen = np.ones(batch_size)

    # During the training of gan,
    # the weights of discriminator should be fixed.
    self.discriminator.trainable = False
    self.generator.trainable = True

    generator_loss = self.model.train_on_batch(noise, y_gen)
    generator_loss = self.model.fit(noise, y_gen, batch_size=batch_size, epochs=1, verbose=0, callbacks=[])

    return {'discriminator_loss': discriminator_loss.history['loss'][0],
            'generator_loss': generator_loss.history['loss'][0]}
