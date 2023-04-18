import numpy as np

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
        backward(dh, dc): Computes the gradients of the LSTM cell parameters and updates them.
        sigmoid(x): Applies the sigmoid function elementwise to an input array.

    """
    def sigmoid(self, x):
        """
        Applies the sigmoid function elementwise to an input array.

        Args:
            x (ndarray): The input array.

        Returns:
            ndarray: The output array after applying the sigmoid function elementwise to x.
        """
        return 1 / (1 + np.exp(-x))

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
            
        # Initialize weights
        self.W_gates = {}
        self.W_gates["input"] = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_gates["output"] = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_gates["forget"] = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.W_candidate = np.random.randn(hidden_size, input_size + hidden_size)
            
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

        
    def backward(self, dh_next, dc_next, dc_top, x_t):
        #TODO
        pass


    def forward(self, x_t):
        # Concatenate input and previous hidden state
        x_t = np.vstack((self.h_t, x_t))
        
        # Compute gates
        z_t = {}
        z_t["input"] = np.dot(self.W_gates["input"], x_t) + self.b_gates["input"]
        z_t["output"] = np.dot(self.W_gates["output"], x_t) + self.b_gates["output"]
        z_t["forget"] = np.dot(self.W_gates["forget"], x_t) + self.b_gates["forget"]
        
        # Compute candidate cell state
        c_t_candidate = np.tanh(np.dot(self.W_candidate, x_t) + self.b_candidate)
        
        # Compute cell state and hidden state
        self.c_t = self.c_t * self.sigmoid(z_t["forget"]) + c_t_candidate * self.sigmoid(z_t["input"])
        self.h_t = np.tanh(self.c_t) * self.sigmoid(z_t["output"])
        
        return 


# Create an instance of LSTM
lstm = LSTM(input_size=2, hidden_size=2, output_size=2)

# Set some input data
x_t = np.array([[1], [2]])

# Call the forward function
lstm.forward(x_t)

# Print the cell state and hidden state
print("Cell state:", lstm.c_t)
print("Hidden state:", lstm.h_t)





