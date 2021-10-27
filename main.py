import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time
import sys, os

def setupCUDA(verbose, device_num):
  """ Setup CUDA Environment to utalise specified GPUs and curtail memory growth"""

  os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)

  physical_devices = tf.config.list_physical_devices('GPU')
  try:
          tf.config.experimental.set_memory_growth(physical_devices[0], True)
  except:

       # Invalid device or cannot modify virtual devices once initialized.
        pass
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  if verbose:
          tf.config.list_physical_devices("GPU")

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, NUM_LATENT_DIM, VAEGAN_LAYER):
    super(CVAE, self).__init__()
    
    self.NUM_LATENT_DIM = NUM_LATENT_DIM #Number of latent dimesions in vae and vaegan bottleneck, and number of input noise values in GAN
    self.VAEGAN_LAYER   = VAEGAN_LAYER   #Only used in VAEGAN mode, layer of disc and enc on which to perform feature wise comparison
    
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(NUM_LATENT_DIM + NUM_LATENT_DIM),
        ]
    )
    
    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(NUM_LATENT_DIM,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )
    
    
    self.discriminator = tf.keras.Sequential(
        [
          tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
          #tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
          #tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(1)
        ]
      )
    
    """
    self.discriminator = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                       input_shape=[28, 28, 1]),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),

      tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1)]

      )
    """
    
    self.generator = tf.keras.Sequential(
      [
        
        
        tf.keras.layers.InputLayer(input_shape=(NUM_LATENT_DIM,)),
        tf.keras.layers.Dense(units=7*7*256, use_bias = False),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(
          filters=128, kernel_size=5, strides=1, padding='same', use_bias = False),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(
          filters=64, kernel_size=5, strides=2, padding='same', use_bias = False),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(
          filters=1, kernel_size=5, strides=2, padding='same', use_bias = False, activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
      ]
    )
    
    self.dis_layer = tf.keras.models.Model(
      inputs=self.discriminator.inputs,
      outputs=self.discriminator.layers[VAEGAN_LAYER].output,
    )

    self.enc_layer = tf.keras.models.Model(
      inputs=self.encoder.inputs,
      outputs=self.encoder.layers[VAEGAN_LAYER].output,
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.NUM_LATENT_DIM))
    return self.decode(eps, apply_sigmoid=True)

  def encode_l(self, x):

    return self.enc_layer(x)

  def discrim_l(self, x):
    
    return self.dis_layer(x)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
  
  def generate(self, z):
    logits = self.generator(z)
    return logits    

  def discriminate(self, x):
    logits = self.discriminator(x)
    return logits

  def discriminator_loss(self, real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

  def generator_loss(self, fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss_gan(model, x):
  
  BATCH_SIZE, NUM_LATENT_DIM = 32, 100
  noise = tf.random.normal([BATCH_SIZE, NUM_LATENT_DIM])
  generated_x = model.generate(noise)
  
  real_output = model.discriminate(x)
  fake_output = model.discriminate(generated_x)
  
  gen_loss = model.generator_loss(fake_output)
  dis_loss = model.discriminator_loss(real_output, fake_output)
  
  return [gen_loss, dis_loss]

def vae(model, x):
  
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  
  global TICKS 
  TICKS = TICKS + 1
  if (TICKS < 5):
    
    print('Test')
    print(logpz, logqz_x)
    
  return x_logit, x, logpz, logqz_x

def compute_loss_vae(model, x):
  
  x_logit, x, logpz, logqz_x = vae(model, x)
  
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  
  return [-tf.reduce_mean(logpx_z + logpz - logqz_x)]

def compute_loss_vaegan(model, x):

  x_logit, x, logpz, logqz_x = vae(model, x)
    
  enc_l = model.encode_l(x)
  dis_l = model.discrim_l(x_logit)
  
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_l, labels=enc_l)
  logpx_z   = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

  real_output = model.discriminate(x)
  fake_output = model.discriminate(x_logit)
  
  gen_loss = model.generator_loss(fake_output)
  dis_loss = model.discriminator_loss(real_output, fake_output)
  
  l_prior = -tf.reduce_mean(logpz - logqz_x)
  l_dis = - tf.reduce_mean(logpx_z) #?
  l_gan   = gen_loss + dis_loss
  
  enc_loss = l_prior + l_dis
  dec_los  = l_dis   + l_gan
  dis_loss = l_gan
  
  return [enc_loss, dec_los, dis_loss]

@tf.function
def train_step_vae(model, x, optimizers, input_size):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    losses = compute_loss_vae(model, x)
    
  gradients = []
  gradients.append(tape.gradient(losses[0], model.trainable_variables))
  
  optimizers[0].apply_gradients(zip(gradients[0], model.trainable_variables))

@tf.function
def train_step_gan(model, x, optimizers, input_size):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      losses = compute_loss_gan(model, x)
    
    gradients = []
    gradients.append(gen_tape.gradient(losses[0], model.generator.trainable_variables))
    gradients.append(disc_tape.gradient(losses[1], model.discriminator.trainable_variables))

    optimizers[0].apply_gradients(zip(gradients[0], model.generator.trainable_variables))
    optimizers[1].apply_gradients(zip(gradients[1], model.discriminator.trainable_variables))
    
@tf.function
def train_step_vaegan(model, x, optimizers, input_size):
  
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
      losses = compute_loss_vaegan(model, x)
    
    gradients = []
    gradients.append(enc_tape.gradient(losses[0], model.encoder.trainable_variables))
    gradients.append(dec_tape.gradient(losses[1], model.decoder.trainable_variables))
    gradients.append(disc_tape.gradient(losses[2], model.discriminator.trainable_variables))

    optimizers[0].apply_gradients(zip(gradients[0], model.encoder.trainable_variables))
    optimizers[1].apply_gradients(zip(gradients[1], model.decoder.trainable_variables))
    optimizers[2].apply_gradients(zip(gradients[2], model.discriminator.trainable_variables))
    
    return 0
  
def generate_and_save_images_vae(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))
  
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
  try:
    os.mkdirs('./tests/vae/')
  except:
    pass

  plt.savefig('./tests/vae/image_at_epoch_{:04d}.png'.format(epoch))
  
def generate_and_save_images_vaegan(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))
  
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
  try:
    os.mkdirs('./tests/vae/')
  except:
    pass

  plt.savefig('./tests/vaegan/image_at_epoch_{:04d}.png'.format(epoch))
  
def generate_and_save_images_gan(model, epoch, test_sample):
  
  NUM_LATENT_DIM =  100
  noise = tf.random.normal([len(test_sample), NUM_LATENT_DIM])
  predictions = model.generate(noise)
    
  fig = plt.figure(figsize=(4, 4))
  
  try:
    os.mkdirs('./tests/vae/')
  except:
    pass

  for i in range(predictions.shape[0]): #predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.savefig('./tests/gan/image_at_epoch_{:04d}.png'.format(epoch))

def main(device_num, mode):
  
  MODE = int(mode)
  setupCUDA(0, device_num)
  
  #Constants:
  NUM_TRAINING_EXAMPLES = 60000
  NUM_TESTING_EXAMPLES  = 10000
  BATCH_SIZE            = 256
  NUM_EPOCHS            = 100
  NUM_LATENT_DIM        = 100
  VAEGAN_LAYER          = 1
  NUM_PLOT              = 16
  
  input_size = [BATCH_SIZE, NUM_LATENT_DIM]

  (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

  train_images = preprocess_images(train_images)
  test_images = preprocess_images(test_images)

  train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
           .shuffle(NUM_TRAINING_EXAMPLES).batch(BATCH_SIZE))
  test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
          .shuffle(NUM_TESTING_EXAMPLES).batch(BATCH_SIZE))
  
  model = CVAE(NUM_LATENT_DIM, VAEGAN_LAYER)

  if (MODE == 0):
    loss_function  = compute_loss_vae
    train_step     = train_step_vae
    learning_rates = [1e-4]
    plot_function  = generate_and_save_images_vae
    
  elif (MODE == 1):
    loss_function  = compute_loss_gan
    train_step     = train_step_gan
    learning_rates = [1e-4, 1e-4]
    plot_function  = generate_and_save_images_gan

  elif (MODE == 2):
    loss_function  = compute_loss_vaegan
    train_step     = train_step_vaegan
    learning_rates = [1e-4, 1e-4, 1e-4]
    plot_function  = generate_and_save_images_vaegan
  
  optimizers = []
  for rate in learning_rates:
    optimizers.append(tf.keras.optimizers.Adam(rate))
    
  # Pick a sample of the test set for generating output images
  assert BATCH_SIZE >= NUM_PLOT
  for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:NUM_PLOT, :, :, :]
  
  plot_function(model, 0, test_sample)
  
  for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    
    for train_x in train_dataset:
        train_step(model, train_x, optimizers, input_size)
      
    end_time = time.time()
  
    loss_objects = []
    for i in range(len(learning_rates)):
      loss_objects.append(tf.keras.metrics.Mean())
    
    for test_x in test_dataset:
      losses = loss_function(model, test_x)
      for i in range(len(learning_rates)):
        loss_objects[i](losses[i])
                
    print('Epoch: {} time elapse for current epoch: {}'.format(epoch, end_time - start_time))
    print('Losses:')
    for i in range(len(learning_rates)):
        print(loss_objects[i].result())

    plot_function(model, epoch, test_sample)

if __name__ == "__main__":
    
    global TICKS
    TICKS = 0
    main(*sys.argv[1:])