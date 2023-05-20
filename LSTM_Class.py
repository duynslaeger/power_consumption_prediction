# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

class AdamOptimizer:
    def __init__(self, parameters, alpha=0.001, beta1=0.09, beta2=0.999, epsilon=1e-8):
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
    
    def __init__(self, hidden_size): 
        self.input_size = 1 # Alsways equal to 1
        self.hidden_size = hidden_size

        # Initialize weights using Xavier initialization
        self.W_gates = {}
        self.W_gates["input"] = np.random.randn(hidden_size, self.input_size + hidden_size) / np.sqrt(self.input_size + hidden_size)
        self.W_gates["output"] = np.random.randn(hidden_size, self.input_size + hidden_size) / np.sqrt(self.input_size + hidden_size)
        self.W_gates["forget"] = np.random.randn(hidden_size, self.input_size + hidden_size) / np.sqrt(self.input_size + hidden_size)
        self.W_candidate = np.random.randn(hidden_size, self.input_size + hidden_size) / np.sqrt(self.input_size + hidden_size)
        
        # Initialize biases with positive forget bias
        self.b_gates = {}
        self.b_gates["input"] = np.random.randn(hidden_size, 1)
        self.b_gates["output"] = np.random.randn(hidden_size, 1)
        self.b_gates["forget"] = np.random.randn(hidden_size, 1)  # Initialized with positive values
        self.b_candidate = np.random.randn(hidden_size, 1)
        
        # Rest of the code remains the same
        self.c_t = np.zeros((hidden_size, 1))
        self.h_t = np.zeros((hidden_size, 1))
        self.dW_gates = {}
        self.dW_gates["input"] = np.zeros((hidden_size, self.input_size + hidden_size))
        self.dW_gates["output"] = np.zeros((hidden_size, self.input_size + hidden_size))
        self.dW_gates["forget"] = np.zeros((hidden_size, self.input_size + hidden_size))
        self.dW_candidate = np.zeros((hidden_size, self.input_size + hidden_size))
        self.db_gates = {}
        self.db_gates["input"] = np.zeros((hidden_size, 1))
        self.db_gates["output"] = np.zeros((hidden_size, 1))
        self.db_gates["forget"] = np.zeros((hidden_size, 1))
        self.db_candidate = np.zeros((hidden_size, 1))
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
        cache : tuple containing : (concat, cprev, gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate)
        """

        concat = np.vstack((x_t, copy.deepcopy(self.h_t)))
        # print(concat)

        gate_forgets = np.dot(self.W_gates["forget"], concat) + self.b_gates["forget"]
        gate_inputs = np.dot(self.W_gates["input"], concat) + self.b_gates["input"]
        gate_outputs = np.dot(self.W_gates["output"], concat) + self.b_gates["output"]
        
        # Apply the sigmoid activation function to the gate values
        f_t = self.sigmoid(gate_forgets)
        i_t = self.sigmoid(gate_inputs)
        o_t = self.sigmoid(gate_outputs)
        
        # Compute the candidate values
        candidate = np.dot(self.W_candidate, concat) + self.b_candidate
        c_candidate = np.tanh(candidate)

        # Keeps a copy of the cell state for next interation
        cprev = copy.deepcopy(self.c_t)

        # Compute the current cell state
        self.c_t = f_t * self.c_t + i_t * c_candidate
        
        # Compute the current hidden state
        self.h_t = o_t * np.tanh(self.c_t)
        # print(o_t)
        # print("c_t\t" + str(self.c_t))
        # print("h_t\t" + str(self.h_t))

        cache = (concat, cprev, gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate)
        
        return cache
        
    def backward(self, dh_t, x_t, cache):
        """
        Computes the gradients of the LSTM cell parameters.

        Args:
        dh_t (numpy array): the gradient of the loss with respect to the hidden state
        x_t (numpy array): the input for the current time step
        cache (tuple) : gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate

        Returns: NOTHING
        """

        concat, cprev, gate_inputs, gate_forgets, gate_outputs,i_t, f_t, o_t, c_candidate = cache
        # Compute the concatenated input and previous hidden state
        dc_t = dh_t * o_t * (1 - np.tanh(self.c_t)**2)

        # Compute the derivatives of the candidate cell state and the input, forget, and output gates
        do_t = dh_t * np.tanh(self.c_t)  * self.dsigmoid(gate_outputs)

        dc_candidate = dc_t * i_t
        df_t = dc_t * cprev * self.dsigmoid(gate_forgets)
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

        # Compute the current cell state
        self.c_t = f_t * self.c_t + i_t * c_candidate
        
        # Compute the current hidden state
        self.h_t = o_t * np.tanh(self.c_t)

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

    def reset(self):
        """
        Réinitialise les gradients des poids et des biais à zéro.
        """
        # Initialize biases with positive forget bias
        self.b_gates = {}
        self.b_gates["input"] = np.zeros((self.hidden_size, 1))
        self.b_gates["output"] = np.zeros((self.hidden_size, 1))
        self.b_gates["forget"] = np.zeros((self.hidden_size, 1))  # Initialized with positive values
        self.b_candidate = np.zeros((self.hidden_size, 1))

        self.dW_gates = {
            "input": np.zeros_like(self.W_gates["input"]),
            "forget": np.zeros_like(self.W_gates["forget"]),
            "output": np.zeros_like(self.W_gates["output"])
        }
        self.db_gates = {
            "input": np.zeros_like(self.b_gates["input"]),
            "forget": np.zeros_like(self.b_gates["forget"]),
            "output": np.zeros_like(self.b_gates["output"])
        }
        self.dW_candidate = np.zeros_like(self.W_candidate)
        self.db_candidate = np.zeros_like(self.b_candidate)

