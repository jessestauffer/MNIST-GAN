import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

class GAN(object):
	def __init__(self, width=28, height=28, color_channels=1):

		self.width = width
		self.height = height
		self.color_channels = color_channels

		self.shape = (self.width, self.height, self.color_channels)
		self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
		
		self.G = self.generator()
		self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

		self.D = self.discriminator()
		self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
		
		self.stacked_generator_discriminator = self.stacked_generator_discriminator()
		self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

	def generator(self):

		model = Sequential()
		model.add(Dense(256, input_shape=(100,)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(self.width  * self.height * self.color_channels, activation='tanh'))
		model.add(Reshape((self.width, self.height, self.color_channels)))

		return model

	def discriminator(self):

		model = Sequential()
		model.add(Flatten(input_shape=self.shape))
		model.add(Dense((self.width * self.height * self.color_channels), input_shape=self.shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense((self.width * self.height * self.color_channels)//2))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))

		return model

	def stacked_generator_discriminator(self):

		self.D.trainable = False

		model = Sequential()
		model.add(self.G)
		model.add(self.D)

		return model

	def train(self, X_train, epochs=20000, batch_size=32, save_interval=100):

		for epoch in range(epochs):

			# train discriminator
			random_index = np.random.randint(0, len(X_train) - batch_size//2)
			legit_images = X_train[random_index : random_index + batch_size//2].reshape(batch_size // 2, self.width, self.height, self.color_channels)

			gen_noise = np.random.normal(0, 1, (batch_size//2, 100))
			synthetic_images = self.G.predict(gen_noise)

			X_combined_batch = np.concatenate((legit_images, synthetic_images))
			y_combined_batch = np.concatenate((np.ones((batch_size//2, 1)), np.zeros((batch_size//2, 1))))

			d_loss = self.D.train_on_batch(X_combined_batch, y_combined_batch)

			# train generator
			noise = np.random.normal(0, 1, (batch_size, 100))
			y_mislabeled = np.ones((batch_size, 1))

			g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabeled)

			print("Epoch: " + str(epoch) + ", Discriminator: " + str(d_loss[0]) + ", Generator: " + str(g_loss))

			if epoch % save_interval == 0:
				self.plot_images(save2file=True, epoch=epoch)

	def plot_images(self, save2file=False, samples=16, epoch=0):
		# plot the generated images
		filename = "images/mnist_" + str(epoch) + ".png"
		noise = np.random.normal(0, 1, (samples, 100))

		images = self.G.predict(noise)

		plt.figure(figsize=(10, 10))

		for i in range(images.shape[0]):
			plt.subplot(4, 4, i+1)
			image = images[i, :, :, :]
			image = np.reshape(image, [self.height, self.width])
			plt.imshow(image, cmap='gray')
			plt.axis('off')
		plt.tight_layout()

		if save2file:
			plt.savefig(filename)
			plt.close('all')
		else:
			plt.show()

if __name__ == '__main__':
	# load the MNIST data
	(X_train, _), (_, _) = mnist.load_data()

	# rescale -1 to 1
	X_train = (X_train - 127.5) / 127.5
	X_train = np.expand_dims(X_train, axis=3)

	gan = GAN()
	gan.train(X_train)