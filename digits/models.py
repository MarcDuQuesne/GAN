from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Sequential

# inherit from sequential?

class Generator:

  def __init__(self):
    super(Discriminator, self).__init__()


    self.model=Sequential()
    self.model.add(Dense(units=256, input_dim=100))
    self.model.add(LeakyReLU(0.2))
    self.model.add(Dense(units=512))
    self.model.add(LeakyReLU(0.2))
    self.model.add(Dense(units=1024))
    self.model.add(LeakyReLU(0.2))
    self.model.add(Dense(units=784, activation='tanh'))

    self.model.compile(loss = 'binary_crossentropy',
                      optimizer = Adam(lr=0.0002, beta_1=0.5))


class Discriminator:

  def __init__(self):

    self.model=Sequential()
    self.model.add(Dense(units = 1024, input_dim = 784))
    self.model.add(LeakyReLU(0.2))
    self.model.add(Dropout(0.3))


    self.model.add(Dense(units = 512))
    self.model.add(LeakyReLU(0.2))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(units=256))
    self.model.add(LeakyReLU(0.2))

    self.model.add(Dense(units=1, activation='sigmoid'))

    self.model.compile(loss = 'binary_crossentropy',
                          optimizer = Adam(lr=0.0002, beta_1=0.5))


class GAN:

  def __init__(self, generator, discriminator):
      # Initialize random noise with generator
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    self.model = Model(inputs = gan_input, outputs = gan_output)

    self.model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam')

  def train_on_batch(self):
    noise= np.random.normal(0,1, [batch_size, 100])

    # Generate fake MNIST images from noised input
    generated_images = self.generator.predict(noise)

    # Get a random set of  real images
    image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

    #Construct different batches of real and fake data
    X= np.concatenate([image_batch, generated_images])

    # Labels for generated and real data
    y_dis=np.zeros(2*batch_size)
    y_dis[:batch_size]=0.9

    #Pre train discriminator on  fake and real data  before starting the gan.
    self.discriminator.model.trainable=True
    self.generator.model.trainable=False

    self.discriminator.model.train_on_batch(X, y_dis)

    #Tricking the noised input of the Generator as real data
    noise = np.random.normal(0,1, [batch_size, 100])
    y_gen = np.ones(batch_size)

    # During the training of gan,
    # the weights of discriminator should be fixed.
    #We can enforce that by setting the trainable flag
    self.discriminator.model.trainable=False
    self.generator.model.trainable=True

    #training  the GAN by alternating the training of the Discriminator
    #and training the chained GAN model with Discriminator’s weights freezed.
    self.model.train_on_batch(noise, y_gen)
