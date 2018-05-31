import numpy as np
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
		model.add(Dense((self.width * self.height * self.color_channels) / 2))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))

		return model

	def stacked_generator_discriminator(self):

		self.D.trainable = False

		model = Sequential()
		model.add(self.G)
		model.add(self.D)

		return model



if __name__ == '__main__':
	# load the MNIST data
	(X_train, _), (_, _) = mnist.load_data()

	# rescale -1 to 1
	X_train = (X_train - 127.5) / 127.5
	X_train = np.expand_dims(X_train, axis=3)