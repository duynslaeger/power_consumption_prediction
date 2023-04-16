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
		-self.weight is a matrix containing the weigths linking each neuron of the current layer to each neuron of the next layer
		-self.bias is a list of the biases linking the current layer to the next one
		"""

		self.weights = np.empty()
		self.biases = np.empty()
		self.next = None
		

	def feed_forward(self, A, , activation_fct):
		"""
		Performs the feeding from the current layer to the next one.
		Returns : a vector containing the activation values of 
				  each neurons within the next layer
		Arguments :
			- A is a vector containing the activation values of each neuron within the layer
			- activation_fct is a string representing the activation function used
		"""


		for b, w in zip(self.biases, self.weights):
			if(activation_fct == 'relu'):
				res = ReLU(np.dot(w, A) + b)

			else:
				print('Please use an accepted activation function. The current accepted activation function are : [relu] ')
				res = np.zeros(b.shape)

		return res


	def add(self, layer):
		"""
		-layer is an object from the Network class 
		"""
		self.next = layer
		

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