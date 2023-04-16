# Authors : Olivier Vertu Ndinga Oba, Bradon Mola, Sacha Duynslaeger
# EPB, ULB

import nn_utils
import numpy as np


# ------------- Neural Network -------------------


class Network(object):
	"""
	Replicate the class Sequential from tensorflow.keras.models
	Creates a neural network with sequential layers
	"""

	def __init__(self, arg):
		"""
		-self.layers is a list containing the layers of the network. A layer is a vector containing the activation values of the layer
		-self.weights is a list of the weights linking two layers. "weights" are matrices. For example, the matrice found at index 0
			of self.weights join the Oth and 1st layer, etc. The format of matrix is used to perform the feed_forward action
		-self.biases is a list of the biases linking two layers. "biases" are vector containing the values of all the biases. For example, the 
			vector found at index 0 of self.biases joint the 0th and 1st layer, etc.
		"""


		# TO DO
		self.layers = np.empty()
		self.weights = np.empty()
		self.biases = np.empty()
		pass

	def feed_forward(self, layer_index, , activation_fct):
		"""
		Performs the feeding from a layer to the next one.
		Returns : a vector containing the activation values of 
				  each neurons within the layer at whose index is layer_index+1
		Arguments :
			- layer_index is the index of the layer that will feed the next layer
			- activation_fct is a string representing the activation function used
		"""


		for b, w in zip(self.biases[layer_index], self.weights[layer_index]):
			if(activation_fct == 'relu'):
				res = ReLU(np.dot(w, self.layers[layer_index]) + b)

			else:
				print('Please use an accepted activation function. The current accepted activation function are : [relu] ')
				res = np.zeros(b.shape)

		return res


	def add(self, layer, weight, biais):
		"""
		"""
		self.layers.append(layer)
		self.weights.append(weight)
		self.biases.append(bias)
		

	def compile():
		# TO DO
		pass

	def fit():
		# TO DO
		pass

	def predict():
		# TO DO
		pass



# ------------- Layers -------------------


def LSTM():
	# TO DO
	pass

def Dense():
	# TO DO
	pass

def InputLayer():
	# TO DO
	pass


# ------------- Utils -------------------

def rmsprop():
	# TO DO
	pass