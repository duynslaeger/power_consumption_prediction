# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
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
        
        for i in range(len(parameters)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            
            # Add missing operations to compute bias-corrected first and second moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            parameters[i] -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)


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
    
    def __init__(self, output_size): 
        self.output_size = output_size 

        # Initialize weights
        self.W_gates = {}
        np.random.seed(10)
        self.W_gates["input"] = np.random.randn(1, 2)* np.sqrt(2 / 2)
        np.random.seed(10)
        self.W_gates["output"] = np.random.randn(1, 2)* np.sqrt(2 / 2)
        np.random.seed(10)
        self.W_gates["forget"] = np.random.randn(1, 2)* np.sqrt(2 / 2)
        np.random.seed(10)
        self.W_candidate = np.random.randn(1, 2)* np.sqrt(2 / 2)
            
        # Initialize biases
        self.b_gates = {}
        self.b_gates["input"] = 0.0 # PAS LE MEME RESULTAT SI ON MET JUSTE 0.0
        self.b_gates["output"] = 0.0
        self.b_gates["forget"] = 0.0
        
        self.b_candidate = 0.0
            
        # Initialize cell state and hidden state
        self.c_t = 0.0
        self.h_t = 0.0
            
        # Initialize gradients
        self.dW_gates = {}
        self.dW_gates["input"] = np.zeros(2)
        self.dW_gates["output"] = np.zeros(2)
        self.dW_gates["forget"] = np.zeros(2)
        
        self.dW_candidate = np.zeros(2) # ERREUR SI ON MET JUSTE 2
            
        self.db_gates = {}
        self.db_gates["input"] = 0.0
        self.db_gates["output"] = 0.0
        self.db_gates["forget"] = 0.0
        
        self.db_candidate = 0.0
        
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
        h_t (numpy array): the hidden state (short term memory) output for the current time step
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

        cache = (gate_inputs, gate_forgets, gate_outputs, candidate, i_t, f_t, o_t, c_candidate)
        
        return cache
        
    def backward(self, dh_t, h_prev, dc_t, c_prev, x_t, cache):
        """
        Computes the gradients of the LSTM cell parameters.

        Args:
        dh_t : the gradient of the loss with respect to the hidden state (short term memory)
        h_prev : the value of the hidden state (short term memory) at the previous time step (t-1)
        dc_t : the gradient of the loss with respect to the cell state (long term memory)
        c_prev : the value of the cell state (long term memory) at the previous time step (t-1)
        x_t : the input for the current time step
        cache (tuple) : gate_inputs, gate_forgets, gate_outputs, candidate, i_t, f_t, o_t, c_candidate

        Returns:
        dW_gates (dictionary): the gradients of the weight matrices for the input, output, and forget gates
        dW_candidate (numpy array): the gradient of the weight matrix for the candidate cell state
        db_gates (dictionary): the gradients of the bias vectors for the input, output, and forget gates
        db_candidate (numpy array): the gradient of the bias vector for the candidate cell state
        """

        gate_inputs, gate_forgets, gate_outputs, candidate, i_t, f_t, o_t, c_candidate = cache
        # Compute the concatenated input and previous hidden state

        z_f = gate_forgets
        z_i = gate_inputs
        z_cand = candidate
        z_o = gate_outputs


        # Applying the first part of the chain rule in order to calculate the elements of the gradient
        # For example : dE/df_t = dE/dh_t * dh_t/dc_t * dc_t/df_t
        dE_dft = dh_t * o_t * (1-np.tanh(self.c_t)**2) * c_prev
        dE_dit = dh_t * o_t * (1-np.tanh(self.c_t)**2) * candidate
        dE_dcandidate = dh_t * o_t * (1-np.tanh(self.c_t)**2) * i_t
        dE_dot = dh_t * np.tanh(self.c_t)

        # Applying the second part of the chain rule but in a way that limitate the number of calculation
        # For example : dE_dwhf = dE/df_t * df_t/dw_hf, with w_hf the weight applied to the hidden state in the forget gate
        prefix_ft = dE_dft * self.dsigmoid(z_f)
        prefix_it = dE_dit * self.dsigmoid(z_i)
        prefix_candidate = dE_dcandidate * (1 - np.tanh(z_cand)**2)
        prefix_ot = dE_dot * self.dsigmoid(z_o)

        # For the following weigths, indice 0 is used to select the weights applied to the input value (x_t), indice 1 to the weights applied to the hidden state (h_t)
        self.dW_gates["forget"][0] += prefix_ft*x_t
        self.dW_gates["forget"][1] += prefix_ft*h_prev
        self.db_gates["forget"] += prefix_ft

        self.dW_gates["input"][0] += prefix_it*x_t
        self.dW_gates["input"][1] += prefix_it*h_prev
        self.db_gates["input"] += prefix_it

        self.dW_candidate[0] += prefix_candidate*x_t
        self.dW_candidate[1] += prefix_candidate*h_prev
        self.db_candidate += prefix_candidate

        self.dW_gates["output"][0] += prefix_ot*x_t
        self.dW_gates["output"][1] += prefix_ot*h_prev
        self.db_gates["output"] += prefix_ot


    def update(self, learning_rate, optimizer=None):
        if optimizer is None:
            for gate in ["input", "output", "forget"]:
                self.W_gates[gate] -= learning_rate * self.dW_gates[gate] # Est-on sur du -= ? J'aurai tendance à plutôt mettre +=
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

#The number of time data you want to use for the prediction
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
# fig, axs = plt.subplots(3, 2, figsize=(12, 12))
# axs = axs.ravel()

# # Plot input data
# axs[0].plot(input_data.flatten())
# axs[0].set_title('Input data')

# # Plot target data
# axs[1].plot(target_data.flatten())
# axs[1].set_title('Target data')

# # Plot input train
# axs[2].plot(input_train.flatten())
# axs[2].set_title('Input train')

# # Plot target train
# axs[3].plot(target_train.flatten())
# axs[3].set_title('Target train')

# # Plot input test
# axs[4].plot(input_test.flatten())
# axs[4].set_title('Input test')

# # Plot target test
# axs[5].plot(target_test.flatten())
# axs[5].set_title('Target test')

# # Show the plot
# plt.tight_layout()
# plt.show()

# Set up the LSTM
lstm = LSTM(output_size=1)

# Train the LSTM
num_epochs = 120
learning_rate = 0.0000001

train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):

    train_loss = 0.0
    val_loss = 0.0

    for i in range(len(input_train)):
        # Get the input and target for this iteration
        
        y_t = target_train[i][0][0]
        for j in range(sequence_length):
            x_t = input_train[i][j]

            # Forward pass
            h_prev = lstm.h_t
            c_prev = lstm.c_t
            cache = lstm.forward(x_t)
            dloss = 2 * (lstm.h_t - y_t)
            # Backward pass
            lstm.backward(dloss, h_prev, 0.0, c_prev, x_t, cache) 
            # Update the weights
            lstm.update(learning_rate, lstm.optimizer)
        
        # Compute the loss and its gradient MAE
        # Compute the loss and its gradient MSE
        loss = (lstm.h_t - y_t) ** 2
        
        train_loss += loss

    # Make predictions on the test set
    lstm_copy = copy.deepcopy(lstm)
    for i in range(len(input_test)):
        y_pred = target_test[i][0][0]
        for j in range(sequence_length):

            x_t = input_test[i][j]

            lstm_copy.forward(x_t)

        val_loss += (lstm_copy.h_t - y_pred) ** 2

    if epoch % 10 == 0:
        print("Epoch", epoch, "training loss", sequence_length*train_loss.flatten()/len(input_train), "Validation loss", sequence_length*val_loss.flatten()/len(input_test))

    # Add the loss values to their respective lists for plotting
    train_loss_list.append(sequence_length*train_loss.flatten()/len(input_train))
    val_loss_list.append(sequence_length*val_loss.flatten()/len(input_test))

# Make predictions on the test set
predictions = []
for i in range(len(input_test)):
    for j in range(sequence_length):

        x_t = input_test[i][j]
        lstm.forward(x_t)
    predictions.append(lstm.h_t)

# Flatten the predictions array
predictions = np.array(predictions).flatten()

# Plot the training and validation loss curves
plt.plot(train_loss_list, label='Training Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.legend()
plt.show()


# Plot the predictions against the actual values
plt.plot(target_test.flatten(), label='Target test')
plt.plot(predictions, label='Predictions on training data')
plt.legend()
plt.show()
