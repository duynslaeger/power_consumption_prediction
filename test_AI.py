# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# Set the random seed


class AdamOptimizer:
    def __init__(self, parameters, alpha=0.00001, beta1=0.9, beta2=0.0999, epsilon=1e-8):
        """
        Initializes the Adam optimizer.

        Args:
            parameters (List[np.ndarray]): A list of numpy arrays representing the parameters to optimize.
            alpha (float): The learning rate used for the update (default: 0.001).
            beta1 (float): The decay rate for the first moment estimate (default: 0.9).
            beta2 (float): The decay rate for the second moment estimate (default: 0.999).
            epsilon (float): A small value used to prevent division by zero (default: 1e-8).
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]
        self.t = 0
    
    def update(self, parameters, gradients):
        """
        Updates the parameters using the Adam optimization algorithm.

        Args:
            parameters (List[np.ndarray]): A list of numpy arrays representing the parameters to optimize.
            gradients (List[np.ndarray]): A list of numpy arrays representing the gradients of the parameters.
        """
        self.t += 1
        alpha_t = self.alpha * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        
        for i in range(len(parameters)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            
            parameters[i] -= alpha_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)


class LSTM:
    """
    Long Short-Term Memory (LSTM) network.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units in the LSTM cell.
        output_size (int): The number of output features.
    
    Attributes:
        W_gates (dict): A dictionary containing weight matrices for the input, output, and forget gates.
        W_candidate (ndarray): The weight matrix for the candidate cell state.
        b_gates (dict): A dictionary containing bias vectors for the input, output, and forget gates.
        b_candidate (ndarray): The bias vector for the candidate cell state.
        c_t (ndarray): The current cell state.
        h_t (ndarray): The current hidden state.
        dW_gates (dict): A dictionary containing gradients of the weight matrices for the input, output, and forget gates.
        dW_candidate (ndarray): The gradient of the weight matrix for the candidate cell state.
        db_gates (dict): A dictionary containing gradients of the bias vectors for the input, output, and forget gates.
        db_candidate (ndarray): The gradient of the bias vector for the candidate cell state.
    
    Methods:
        forward(x): Performs the forward pass through the LSTM cell.
        backward(dh, dc, x, cache): Computes the gradients of the LSTM cell parameters 
        updates updates the gradients of the LSTM
        sigmoid(x): Applies the sigmoid function elementwise to an input array.
        dsigmoid(self, x)

    """
    
    def __init__(self, input_size, hidden_size, output_size): 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size 

        # Initialize weights
        self.W_gates = {}
        np.random.seed(10)
        self.W_gates["input"] = np.random.randn(hidden_size, input_size + hidden_size)* np.sqrt(2 / (input_size + hidden_size))
        np.random.seed(10)
        self.W_gates["output"] = np.random.randn(hidden_size, input_size + hidden_size)* np.sqrt(2 / (input_size + hidden_size))
        np.random.seed(10)
        self.W_gates["forget"] = np.random.randn(hidden_size, input_size + hidden_size)* np.sqrt(2 / (input_size + hidden_size))
        np.random.seed(10)
        self.W_candidate = np.random.randn(hidden_size, input_size + hidden_size)* np.sqrt(2 / (input_size + hidden_size))
            
        # Initialize biases
        self.b_gates = {}
        self.b_gates["input"] = np.zeros((hidden_size, 1))
        self.b_gates["output"] = np.zeros((hidden_size, 1))
        self.b_gates["forget"] = np.zeros((hidden_size, 1))
        
        self.b_candidate = np.zeros((hidden_size, 1))
            
        # Initialize cell state and hidden state
        self.c_t = np.zeros((hidden_size, 1))
        self.h_t = np.zeros((hidden_size, 1))
            
        # Initialize gradients
        self.dW_gates = {}
        self.dW_gates["input"] = np.zeros((hidden_size, input_size + hidden_size))
        self.dW_gates["output"] = np.zeros((hidden_size, input_size + hidden_size))
        self.dW_gates["forget"] = np.zeros((hidden_size, input_size + hidden_size))
        
        self.dW_candidate = np.zeros((hidden_size, input_size + hidden_size))
            
        self.db_gates = {}
        self.db_gates["input"] = np.zeros((hidden_size, 1))
        self.db_gates["output"] = np.zeros((hidden_size, 1))
        self.db_gates["forget"] = np.zeros((hidden_size, 1))
        
        self.db_candidate = np.zeros((hidden_size, 1))
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(parameters=[self.W_gates["input"], self.W_gates["output"], self.W_gates["forget"], 
                                            self.b_gates["input"], self.b_gates["output"], self.b_gates["forget"], 
                                            self.W_candidate, self.b_candidate])

    def sigmoid(self, x):
        """
        Applies the sigmoid function elementwise to an input array.

        Args:
            x (ndarray): The input array.

        Returns:
            ndarray: The output array after applying the sigmoid function elementwise to x.
        """

        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        """
        Computes the derivative of the sigmoid function elementwise for an input array.

        Args:
            x (ndarray): The input array.

        Returns:
            ndarray: The output array after applying the derivative of the sigmoid function elementwise to x.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    def forward(self, x_t):
        """
        Performs the forward pass through the LSTM.

        Args:
        x_t (numpy array): the input for the current time step

        Returns:
        h_t (numpy array): the hidden state output for the current time step
        self.c_t (numpy array):
        """
        # Concatenate the previous hidden state and the current input
        concat = np.vstack((x_t, self.h_t))
        
        # Compute the input, forget, and output gate values
        gate_inputs = np.dot(self.W_gates["input"], concat) + self.b_gates["input"]
        gate_forgets = np.dot(self.W_gates["forget"], concat) + self.b_gates["forget"]
        gate_outputs = np.dot(self.W_gates["output"], concat) + self.b_gates["output"]
        
        # Apply the sigmoid activation function to the gate values
        i_t = self.sigmoid(gate_inputs)
        f_t = self.sigmoid(gate_forgets)
        o_t = self.sigmoid(gate_outputs)
        
        # Compute the candidate values
        candidate = np.dot(self.W_candidate, concat) + self.b_candidate
        c_candidate = np.tanh(candidate)
        
        # Compute the current cell state
        self.c_t = f_t * self.c_t + i_t * c_candidate
        
        # Compute the current hidden state
        self.h_t = o_t * np.tanh(self.c_t)

        cache = (gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate)
        
        return cache
        
    def backward(self, dh_t, dc_t, x_t, cache):
        """
        Computes the gradients of the LSTM cell parameters.

        Args:
        dh_t (numpy array): the gradient of the loss with respect to the hidden state
        dc_t (numpy array): the gradient of the loss with respect to the cell state
        x_t (numpy array): the input for the current time step
        cache (tuple) : gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate

        Returns:
        dX_t (numpy array): the gradient of the loss with respect to the input at the current time step
        dW_gates (dictionary): the gradients of the weight matrices for the input, output, and forget gates
        dW_candidate (numpy array): the gradient of the weight matrix for the candidate cell state
        db_gates (dictionary): the gradients of the bias vectors for the input, output, and forget gates
        db_candidate (numpy array): the gradient of the bias vector for the candidate cell state
        """

        gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate = cache
        # Compute the concatenated input and previous hidden state
        concat = np.vstack((x_t, self.h_t))

        # Compute the derivatives of the candidate cell state and the input, forget, and output gates
        dc_candidate = dh_t * o_t * (1 - np.tanh(self.c_t)**2)
        do_t = dh_t * np.tanh(self.c_t) * self.dsigmoid(gate_outputs)
        df_t = dc_t * self.c_t * self.dsigmoid(gate_forgets)
        di_t = dc_t * c_candidate * self.dsigmoid(gate_inputs)

        # Compute the gradients of the weight matrices and bias vectors for the input, output, and forget gates
        self.dW_gates["input"] += np.dot(di_t, concat.T)
        self.dW_gates["forget"] += np.dot(df_t, concat.T)
        self.dW_gates["output"] += np.dot(do_t, concat.T)
        self.dW_candidate += np.dot(dc_candidate, concat.T)

        self.db_gates["input"] += di_t
        self.db_gates["forget"] += df_t
        self.db_gates["output"] += do_t
        self.db_candidate += dc_candidate

        # Compute the gradient of the loss with respect to the input at the current time step
        dX_t = np.dot(self.W_gates["input"].T, di_t) + np.dot(self.W_gates["forget"].T, df_t) + np.dot(self.W_gates["output"].T, do_t) + np.dot(self.W_candidate.T, dc_candidate)

        # Compute the gradient of the loss with respect to the previous cell state and hidden state
        dc_t_prev = dc_t * f_t
        dh_t_prev = np.dot(self.W_gates["input"].T, di_t) + np.dot(self.W_gates["forget"].T, df_t) + np.dot(self.W_gates["output"].T, do_t)

        # Update the current cell state and hidden state
        self.c_t = self.c_t * f_t + c_candidate * i_t
        self.h_t = np.tanh(self.c_t) * o_t

        return dX_t

    def update(self, learning_rate, optimizer=None):
        if optimizer is None:
            for gate in ["input", "output", "forget"]:
                self.W_gates[gate] -= learning_rate * self.dW_gates[gate]
                self.b_gates[gate] -= learning_rate * self.db_gates[gate]

            self.W_candidate -= learning_rate * self.dW_candidate
            self.b_candidate -= learning_rate * self.db_candidate
        else:
            optimizer.update(parameters=[self.W_gates["input"], self.W_gates["output"], self.W_gates["forget"], 
                                          self.b_gates["input"], self.b_gates["output"], self.b_gates["forget"], 
                                          self.W_candidate, self.b_candidate],
                              gradients=[self.dW_gates["input"], self.dW_gates["output"], self.dW_gates["forget"], 
                                         self.db_gates["input"], self.db_gates["output"], self.db_gates["forget"], 
                                         self.dW_candidate, self.db_candidate])

#You need to choose the sequence size which should be the same as the input size.
sequence_length = 1
#You also need to choose the prediction size which should be the same as the hidden size.
predict_size = 1

import numpy as np
import matplotlib.pyplot as plt

# Took a data 
dataset = pd.read_csv('CDB005_15min.csv', usecols=['ts', 'p_cons'], index_col='ts', parse_dates=['ts'])

# Set up the input and target data for the LSTM

# Add p_cons data to input_data
input_data = []
target_data = []
for i in range(len(dataset) - sequence_length - predict_size):
    input_data.append(np.array([dataset['p_cons'][i:i+sequence_length]]))
    target_data.append(dataset['p_cons'][i+sequence_length : i+sequence_length + predict_size])
input_data = np.array(input_data).reshape(len(input_data), sequence_length, 1)
target_data = np.array(target_data).reshape(len(target_data), predict_size, 1)

# Normalize the data using min-max normalization
input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
target_data = (target_data - np.min(target_data)) / (np.max(target_data) - np.min(target_data))

# Split the data into training and testing sets
num_samples = len(input_data)
num_training_samples = int(num_samples * 0.75)

input_train = input_data[:num_training_samples]
target_train = target_data[:num_training_samples]
input_test = input_data[num_training_samples:]
target_test = target_data[num_training_samples:]

import matplotlib.pyplot as plt

# Define the size of the figure
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.ravel()

# Plot input data
axs[0].plot(input_data.flatten())
axs[0].set_title('Input data')

# Plot target data
axs[1].plot(target_data.flatten())
axs[1].set_title('Target data')

# Plot input train
axs[2].plot(input_train.flatten())
axs[2].set_title('Input train')

# Plot target train
axs[3].plot(target_train.flatten())
axs[3].set_title('Target train')

# Plot input test
axs[4].plot(input_test.flatten())
axs[4].set_title('Input test')

# Plot target test
axs[5].plot(target_test.flatten())
axs[5].set_title('Target test')

# Show the plot
plt.tight_layout()
plt.show()

# Set up the LSTM
lstm = LSTM(input_size=sequence_length, hidden_size=predict_size, output_size=1)

# Train the LSTM
num_epochs = 120
learning_rate = 0.0000001
for epoch in range(num_epochs):
    for i in range(len(input_train)):
        # Get the input and target for this iteration
        x_t = input_train[i]
        y_t = target_train[i]
        
        # Forward pass
        cache = lstm.forward(x_t)
        
        # Compute the loss and its gradient MAE
        # Compute the loss and its gradient MSE
        loss = np.sum((lstm.h_t - y_t) ** 2)
        dloss = 2 * (lstm.h_t - y_t)
        
        # Backward pass
        lstm.backward(dloss, np.zeros((lstm.hidden_size, 1)), x_t, cache)
        
        # Update the weights
        lstm.update(learning_rate, lstm.optimizer)
        
    # Print the loss every 10 epochs
    if epoch % 10 == 0:
        print("Epoch", epoch, "Loss", loss)

# Make predictions on the test set
predictions = []
for i in range(len(input_test)):
    x_t = input_test[i]
    lstm.forward(x_t)
    predictions.append(lstm.h_t)

# Flatten the predictions array
predictions = np.array(predictions).flatten()

# Plot the predictions against the actual values
plt.plot(target_test.flatten(), label='Target test')
plt.plot(predictions, label='Predictions on training data')
plt.legend()
plt.show()
