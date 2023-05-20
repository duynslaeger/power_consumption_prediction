from LSTM_Class import *
from train import preprocess_sequential_data
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import argparse



def read_weights_biases_from_file(file_path):
    weights_biases = {}
    current_variable = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Check for variable headers
            if line.endswith(":"):
                current_variable = line[:-1]
                weights_biases[current_variable] = []
            else:

                elements = line.strip('[]').split()
                # Convert the elements to floats and reshape into a matrix
                float_matrix = np.array([float(element) for element in elements]).reshape((1, -1))
                weights_biases[current_variable] = float_matrix

    return weights_biases


def assign_weights_biases(lstm, weights_biases):
    lstm.W_gates["input"] = weights_biases["W_gates[input]"]
    lstm.W_gates["output"] = weights_biases["W_gates[output]"]
    lstm.W_gates["forget"] = weights_biases["W_gates[forget]"]
    lstm.W_candidate = weights_biases["W_candidate"]
    lstm.b_gates["input"] = weights_biases["b_gates[input]"]
    lstm.b_gates["output"] = weights_biases["b_gates[output]"]
    lstm.b_gates["forget"] = weights_biases["b_gates[forget]"]
    lstm.b_candidate = weights_biases["b_candidate"]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=5, help='Length of the sequence')
    parser.add_argument('--pred_size', type=int, default=1, help='Size of the prediction')
    parser.add_argument('--file_path', type=str, default='Data/CDB002.csv', help='Path to the file')
    parser.add_argument('--parameters_file_path', type=str, default='Saved_parameters/weights_biases.txt', help='Path to the text file')

    args = parser.parse_args()

    # The number of time data you want to use for the prediction
    sequence_length = args.seq_len
    # You also need to choose the prediction size which should be the same as the hidden size.
    predict_size = args.pred_size
    # The path of the file you want to predict from
    file_path = args.file_path

    # Path to the trained weights and biases file
    param_file_path = args.parameters_file_path

    weights_biases = read_weights_biases_from_file(param_file_path)

    lstm = LSTM(hidden_size=predict_size)
    # Assign the weights and biases
    assign_weights_biases(lstm, weights_biases)

    # Pre-process the data
    data_train, data_val, data_test = preprocess_sequential_data('Data/CDB002.csv', sequence_length, predict_size)

    # TEST TEST TEST with DataTrain

    # predictions = data_train[:sequence_length].tolist()

    # print(predictions)
    # # Make predictions on the test set
    # for i in range(len(data_train) - sequence_length - predict_size):
    #     for j in range(sequence_length):
    #         x_t = predictions[i+j]
        
    #         cache = lstm.forward(x_t)

    #     for n in range(predict_size):
    #         predictions.append(lstm.h_t[n])

    # print(predictions)

    predictions = data_test[:sequence_length].tolist()

    # Make predictions on the test set
    for i in range(len(data_test) - sequence_length - predict_size):
        for j in range(sequence_length):
            x_t = predictions[i+j]
        
            cache = lstm.forward(x_t)

        for n in range(predict_size):
            predictions.append(lstm.h_t[n])


    # Plot the predictions against the actual values
    plt.plot(data_test[sequence_length:], label='Expected value')
    plt.plot(predictions[sequence_length:], label='Predictions on training data')
    plt.legend()
    plt.show()