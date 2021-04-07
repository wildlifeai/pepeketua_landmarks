import tensorflow as tf
import tensorflow.keras as keras

class CNNBlock(keras.layers.Layer):
	"""docstring for CNNBlock"""
	def __init__(self, num_layers, num_filters, **kwargs):
		super(CNNBlock, self).__init__(**kwargs)
		self.hidden = [keras.layers.Conv2D(num_filters, (3,3), activation = "relu")
						for _ in range(num_layers)]

	def call(self, inputs):
		x = inputs
		for layer in self.hidden:
			x = layer(x)

		return x
		