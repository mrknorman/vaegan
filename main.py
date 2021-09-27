import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time

def setupCUDA(verbose, device_num):

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

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1], activation = tf.keras.layers.ReLU(), name = "test_layer"),
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = tf.keras.layers.ReLU()),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
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

    self.dis_layer = tf.keras.models.Model(
      inputs=self.discriminator.inputs,
      outputs=self.discriminator.layers[0].output,
    )

    self.enc_layer = tf.keras.models.Model(
      inputs=self.encoder.inputs,
      outputs=self.encoder.layers[0].output,
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
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

def compute_loss_vae(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def vae(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  enc_l = model.encode_l(x)
  dis_l = model.discrim_l(x_logit)

  #print(f"Encoder: {enc_l[0][0][0]}")
  #print(f"Discriminator: {dis_l[0][0][0]}")

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_l, labels=enc_l)
  logpx_z   = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)

  return x_logit, logpx_z, logpz, logqz_x

def compute_loss(model, x):

  generated_x, logpx_z, logpz, logqz_x = vae(model, x);

  real_output = model.discriminate(x)
  fake_output = model.discriminate(generated_x)
  
  gen_loss = model.generator_loss(fake_output)
  dis_loss = model.discriminator_loss(real_output, fake_output)

  return -tf.reduce_mean(logpx_z + logpz - logqz_x) + (dis_loss - gen_loss)

@tf.function
def train_step(model, x, optimizer, loss_function):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = loss_function(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_and_save_images(model, epoch, test_sample, name):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('./tests/{}/image_at_epoch_{:04d}.png'.format(name, epoch))

def main():
  
  train_size = 60000
  batch_size = 32
  test_size  = 10000

  (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

  train_images = preprocess_images(train_images)
  test_images = preprocess_images(test_images)

  train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
           .shuffle(train_size).batch(batch_size))
  test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
          .shuffle(test_size).batch(batch_size))

  epochs = 10
  # set the dimensionality of the latent space to a plane for visualization later
  latent_dim = 2
  num_examples_to_generate = 16

  optimizer = tf.keras.optimizers.Adam(1e-4)

  # keeping the random vector constant for generation (prediction) so
  # it will be easier to see the improvement.
  random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
  model = CVAE(latent_dim)
  #model.build(input_shape = (:,))
  # Pick a sample of the test set for generating output images
  assert batch_size >= num_examples_to_generate
  for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]

  is_vae = 0

  if is_vae:
    loss_function = compute_loss_vae
    plot_name     = "vae"
  else:
    loss_function = compute_loss
    plot_name     = "vaegan"

  generate_and_save_images(model, 0, test_sample, plot_name)

  for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer, loss_function)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(loss_function(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
      .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample, plot_name)



if __name__ == "__main__":
    main()