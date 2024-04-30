import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time
import sys, os

from sklearn import manifold
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from numpy import random
import string

print(tf.executing_eagerly())

#Variables:
MODEL_TYPE            = 3 #<-- Model Type to Employ (0 = Standard VAE, 1 = Standard GAN, 2 = VAE/GAN, 3 = ADVAE)
DEVICE_NUM            = 1 #<-- GPU Number to use, ignore on COLAB.
NUM_TRAINING_EXAMPLES = 60000 #<-- Number of training examples.
NUM_TESTING_EXAMPLES  = 10000 #<-- Number of testing examples.
BATCH_SIZE            = 256   #<-- Batch Size
NUM_EPOCHS            = 100   #<-- Number of training epochs
NUM_LATENT_DIM        = 100   #<-- Latent Space Size
VAEGAN_LAYER          = 1     #<-- Used in VAE/GAN -- layer to compare latent space
NUM_PLOT              = 16    #<-- Number of examples to plot at each epoch

#Derived Parameters:
input_size = [BATCH_SIZE, NUM_LATENT_DIM]

discriminators = {"VAE": 
                    tf.keras.Sequential(
                    [
                      tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                       input_shape=[28, 28, 1]),
                      tf.keras.layers.LeakyReLU(),
                      tf.keras.layers.Dropout(0.3),

                      tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                      tf.keras.layers.LeakyReLU(),
                      tf.keras.layers.Dropout(0.3),

                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(1)
                    ]),
                  "GAN": 
                    tf.keras.Sequential(
                    [
                      tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                       input_shape=[28, 28, 1]),
                      tf.keras.layers.LeakyReLU(),
                      tf.keras.layers.Dropout(0.3),

                      tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                      tf.keras.layers.LeakyReLU(),
                      tf.keras.layers.Dropout(0.3),

                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(1)
                    ]), 
                  "VAEGAN":
                    tf.keras.Sequential(
                    [
                      tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                 input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
                      tf.keras.layers.Dropout(0.3),
                      tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
                      tf.keras.layers.Dropout(0.3),
                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(1)
                    ]),
                    "ADVAE":
                    tf.keras.Sequential(
                    [
                      tf.keras.layers.Dense(128, input_shape=[NUM_LATENT_DIM],),
                      tf.keras.layers.Dense(64),
                      tf.keras.layers.Dense(1)
                    ])
                  }

encoders       = {"VAE": 
                        tf.keras.Sequential(
                        [
                            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                      input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
                            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
                            tf.keras.layers.Flatten(),
                            # No activation
                            tf.keras.layers.Dense(NUM_LATENT_DIM + NUM_LATENT_DIM),
                        ]
                        ),
                  "GAN": 
                    tf.keras.Sequential(
                        [
                            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                      input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
                            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
                            tf.keras.layers.Flatten(),
                            # No activation
                            tf.keras.layers.Dense(NUM_LATENT_DIM + NUM_LATENT_DIM),
                        ]
                        ), 
                  "VAEGAN":
                    tf.keras.Sequential(
                        [
                            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                      input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
                            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
                            tf.keras.layers.Flatten(),
                            # No activation
                            tf.keras.layers.Dense(NUM_LATENT_DIM + NUM_LATENT_DIM),
                        ]
                        ),
                    "ADVAE":
                    tf.keras.Sequential(
                        [
                            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                      input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
                            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
                            tf.keras.layers.Flatten(),
                            # No activation
                            tf.keras.layers.Dense(NUM_LATENT_DIM),
                        ]
                        )
                  }

decoders       = {"VAE": tf.keras.Sequential(
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
                        ]),
                      "GAN": 
                    
                        tf.keras.Sequential([
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
                        ]), 
                  "VAEGAN":
                        tf.keras.Sequential([
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
                        ]),
                    "ADVAE":
                    tf.keras.Sequential(
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
                  }

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
			
def calculate_real_loss(data):
  return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)(tf.ones_like(data), data)

def calculate_fake_loss(data):
  return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)(tf.zeros_like(data), data)

def gan(real_data, fake_data):

  real_loss = calculate_real_loss(real_data)
  fake_losses = []

  for data in fake_data:
    fake_losses.append(calculate_fake_loss(data))

  return [np.sum(fake_losses), real_loss]

def compute_loss_gan(model, x):
  
  noise = tf.random.normal([BATCH_SIZE, NUM_LATENT_DIM])
  generated_x = model.generate(noise)
  
  real_output = model.discriminate(x)
  fake_output = model.discriminate(generated_x)
  
  return [model.generator_loss(fake_output), model.discriminator_loss(real_output, fake_output)]

def vae(model, x):
  
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  
  logpz   = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
    
  return x_logit, x, logpz, logqz_x

def compute_loss_vae(model, x):
  
  x_logit, x, logpz, logqz_x = vae(model, x)
  
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  
  return [-tf.reduce_mean(logpx_z + logpz - logqz_x)]

def compute_loss_vaegan(model, x):

  x_logit, x, logpz, logqz_x = vae(model, x)
    
  real_features = model.discrim_l(x)
  fake_features = model.discrim_l(x_logit)
  
  logpx_z       = tf.reduce_sum((fake_features-real_features)**2)

  noise         = tf.random.normal([x.shape[0], NUM_LATENT_DIM])

  real_output   = model.discriminate(x)
  fake_x_output = model.discriminate(x_logit)
  fake_z_output = model.discriminate(model.decode(noise))

  l_prior       = -tf.reduce_sum(logpz - logqz_x)
  l_dis         = logpx_z
  l_gan         = np.sum(gan(real_output, [fake_x_output, fake_z_output]))
  
  enc_loss      = l_prior + l_dis
  dec_los       = l_dis   - l_gan
  dis_loss      = l_gan
  
  return [enc_loss, dec_los, dis_loss]

def compute_loss_advae(model, x):

  fake_z = model.encode_(x)
  
  x_logit = model.decode(fake_z)
    
  cross_ent = tf.reduce_sum((x-x_logit)**2)

  z      = tf.convert_to_tensor(np.random.normal(size = fake_z.shape))
  
  real_output = model.discriminate(z)
  fake_output = model.discriminate(fake_z) 
  
  l_reconstruct = cross_ent
  l_regularise  = np.sum(gan(real_output, [fake_output]))

  return [l_reconstruct, l_regularise]

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
      
      #tf.clip_by_value(losses[0], 0, 100000);
    
    gradients = []
    gradients.append(enc_tape.gradient(losses[0], model.encoder.trainable_variables))
    gradients.append(dec_tape.gradient(losses[1], model.decoder.trainable_variables))
    gradients.append(disc_tape.gradient(losses[2], model.discriminator.trainable_variables))

    optimizers[0].apply_gradients(zip(gradients[0], model.encoder.trainable_variables))
    optimizers[1].apply_gradients(zip(gradients[1], model.decoder.trainable_variables))
    optimizers[2].apply_gradients(zip(gradients[2], model.discriminator.trainable_variables))
    
    return 0
  
@tf.function
def train_step_advae(model, x, optimizers, input_size):
  
    with tf.GradientTape() as reconstruct_tape, tf.GradientTape() as regularise_tape:
      losses = compute_loss_advae(model, x)
    
    gradients = []
    gradients.append(reconstruct_tape.gradient(losses[0], model.encoder.trainable_variables + model.decoder.trainable_variables))
    gradients.append(regularise_tape.gradient(losses[1], model.encoder.trainable_variables + model.discriminator.trainable_variables))

    optimizers[0].apply_gradients(zip(gradients[0], model.encoder.trainable_variables + model.decoder.trainable_variables))
    optimizers[1].apply_gradients(zip(gradients[1], model.encoder.trainable_variables + model.discriminator.trainable_variables))
    
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def generate_and_save_images_vae(model, epoch, test_sample, reconstruct = None):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)

  if (reconstruct != None):
    z = None;

  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))
  
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
  try:
    os.mkdirs('./vegan_tests/tests/vae/')
  except:
    pass

  plt.savefig('./vegan_tests/tests/vae/image_at_epoch_{:04d}.png'.format(epoch))

def generate_and_save_images_advae(model, epoch, test_sample, reconstruct = None):

  z = model.encode_(test_sample)
  
  if (reconstruct != None):
    z = None;
  
  predictions = model.sample(z)

  fig = plt.figure(figsize=(4, 4))
  
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
  try:
    os.mkdirs('./vegan_tests/tests/advae/')
  except:
    pass

  plt.savefig('./vegan_tests/tests/advae/image_at_epoch_{:04d}.png'.format(epoch))
  
def generate_and_save_images_vaegan(model, epoch, test_sample, reconstruct = None):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  
  if (reconstruct != None):
    z = None;
  
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))
  
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
  try:
    os.mkdirs('./vegan_tests/tests/vae/')
  except:
    pass

  plt.savefig('./vegan_tests/tests/vaegan/image_at_epoch_{:04d}.png'.format(epoch))
  
def generate_and_save_images_gan(model, epoch, test_sample):
  
  NUM_LATENT_DIM =  100
  noise = tf.random.normal([len(test_sample), NUM_LATENT_DIM])
  predictions = model.generate(noise)
    
  fig = plt.figure(figsize=(4, 4))
  
  try:
    os.mkdirs('./vegan_tests/tests/gan/')
  except:
    pass

  for i in range(predictions.shape[0]): #predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.savefig('./vegan_tests/tests/gan/image_at_epoch_{:04d}.png'.format(epoch))
  
MODE = int(MODEL_TYPE)
setupCUDA(1, DEVICE_NUM)

tf.random.set_seed(123489)

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images     = preprocess_images(train_images)
test_images      = preprocess_images(test_images)

train_dataset    = (tf.data.Dataset.from_tensor_slices(train_images)
         .shuffle(NUM_TRAINING_EXAMPLES).batch(BATCH_SIZE))
test_dataset     = (tf.data.Dataset.from_tensor_slices(test_images)
         .shuffle(NUM_TESTING_EXAMPLES).batch(BATCH_SIZE))
  
if (MODE == 0):
  name           = "VAE"
  loss_function  = compute_loss_vae
  train_step     = train_step_vae
  learning_rates = [1e-4]
  plot_function  = generate_and_save_images_vae
    
elif (MODE == 1):
  name           = "GAN"
  loss_function  = compute_loss_gan
  train_step     = train_step_gan
  learning_rates = [1e-4, 1e-4]
  plot_function  = generate_and_save_images_gan

elif (MODE == 2):
  name           = "VAEGAN"
  loss_function  = compute_loss_vaegan
  train_step     = train_step_vaegan
  learning_rates = [1e-4, 1e-4, 1e-4]
  plot_function  = generate_and_save_images_vaegan
    
elif (MODE == 3):
  name           = "ADVAE"
  loss_function  = compute_loss_advae
  train_step     = train_step_advae
  learning_rates = [1e-4, 1e-4]
  plot_function  = generate_and_save_images_advae
  
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, NUM_LATENT_DIM, VAEGAN_LAYER, NAME):
    super(CVAE, self).__init__()
    
    self.NUM_LATENT_DIM = NUM_LATENT_DIM #Number of latent dimesions in vae and vaegan bottleneck, and number of input noise values in GAN
    self.VAEGAN_LAYER   = VAEGAN_LAYER   #Only used in VAEGAN mode, layer of disc and enc on which to perform feature wise comparison
    
    self.encoder       = encoders      [NAME]
    self.decoder       = decoders      [NAME]
    self.discriminator = discriminators[NAME]
    
    self.generator = tf.keras.Sequential(
      [ 
        tf.keras.layers.InputLayer(input_shape=(NUM_LATENT_DIM,)),
        tf.keras.layers.Dense(units=7*7*256, use_bias = False),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(
          filters=128, kernel_size=5, strides=1, padding='same', use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(
          filters=64, kernel_size=5, strides=2, padding='same', use_bias = False),
        tf.keras.layers.BatchNormalization(),
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
      eps = tf.random.normal(shape=(NUM_PLOT, self.NUM_LATENT_DIM))
    return self.decode(eps, apply_sigmoid=True)

  def encode_l(self, x):

    return self.enc_layer(x)

  def discrim_l(self, x):
    
    return self.dis_layer(x)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def encode_(self, x):
    return self.encoder(x)

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
  
model = CVAE(NUM_LATENT_DIM, VAEGAN_LAYER, name)

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
  
plot_function(model, 100, test_sample, reconstruct = True)
plt.savefig("reconstruction.png")

plot_function(model, 100, test_sample)
plt.savefig("random_samples.png")

def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i, img in enumerate(imageData):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = img.reshape([img.shape[0],img.shape[1]])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

def computeTSNEProjectionOfLatentSpace(X, encoder, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder.predict(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne
"""
X_tsne = computeTSNEProjectionOfLatentSpace(test_images, model.encoder, display = False)

r_test_images = test_images[1:200]
r_X_tsne = X_tsne[1:200]

print("Plotting t-SNE visualization...")
fig, ax = plt.subplots()
imscatter(r_X_tsne[:, 0], r_X_tsne[:, 1], imageData=r_test_images, ax=ax, zoom=0.6)
plt.savefig("latent_space.png")
"""

def visualizeInterpolation(start, end, model, encode, decode, save=False, nbSteps=5, num = 0):
    print("Generating interpolations...")

    # Create micro batch
    X = np.array([start,end])
    
    vae = 0;
    if vae: 
      # Compute latent space projection
      mean, logvar = model.encode(np.array([start]))
      latentStart = model.reparameterize(mean, logvar)

      mean, logvar = model.encode(np.array([end]))
      latentEnd = model.reparameterize(mean, logvar)
    else:
      latentStart = model.encode_(np.array([start]))
      latentEnd = model.encode_(np.array([end]))

    # Get original image for comparison
    startImage, endImage = X

    normalImages = []
    #Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)

    reconstructions = []
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart*(1-alpha) + latentEnd*alpha
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage,1-alpha,endImage,alpha,0)
        normalImages.append(blendImage)
        
        decoded = model.decode(vector, apply_sigmoid = True).numpy()
        decoded = decoded - np.min(decoded)
        reconstructions.append(np.multiply(decoded, 1.0/np.max(decoded)))

    np.array(reconstructions)
    # Put final image together
    resultLatent = None
    resultImage = None

    if save:
        hashName = num

    for i in range(len(reconstructions)):
      
        interpolatedImage = normalImages[i]*255
        interpolatedImage = cv2.resize(interpolatedImage,(50,50))
        interpolatedImage = interpolatedImage.astype(np.uint8)
        resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage,interpolatedImage])

        reconstructedImage = reconstructions[i]*255.
        reconstructedImage = reconstructedImage.reshape([28,28])
        reconstructedImage = cv2.resize(reconstructedImage,(50,50))
        reconstructedImage = reconstructedImage.astype(np.uint8)
        resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent,reconstructedImage])
    
        result = np.vstack([resultImage,resultLatent])
      
    if save:
        cv2.imwrite("{}_{}.png".format(hashName,i),result)

    if not save:
        cv2.imshow(result)
        
print(np.argmax(test_images[0]), np.argmax(test_images[3]))


visualizeInterpolation((test_images[56]/np.max(test_images[56])), (test_images[87]/np.max(test_images[87])), model, model.encode_, model.decoder, save=True, nbSteps=5, num = 0)
visualizeInterpolation((test_images[1]/np.max(test_images[1])), (test_images[9]/np.max(test_images[9])), model, model.encode_, model.decoder, save=True, nbSteps=5, num = 1)
visualizeInterpolation((test_images[0]/np.max(test_images[0])), (test_images[7]/np.max(test_images[7])), model, model.encode_, model.decoder, save=True, nbSteps=5, num = 2)
visualizeInterpolation((test_images[4]/np.max(test_images[4])), (test_images[21]/np.max(test_images[21])), model, model.encode_, model.decoder, save=True, nbSteps=5, num = 3)