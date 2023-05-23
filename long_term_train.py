from LSTM_Class import *
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import argparse

def train_lstm(lstm, data_train, data_val, sequence_length, predict_size, num_epochs, learning_rate, compute_validation=False):
    train_loss_list = []
    val_loss_list = []


    for epoch in range(num_epochs):
        train_predictions = []
        val_predictions = []
        for s in range(sequence_length):
            train_predictions.append(np.array(data_train[s]))
            val_predictions.append(np.array(data_val[s]))
        train_loss = 0.0
        val_loss = 0.0

        for i in range(0, len(data_train) - sequence_length - predict_size, predict_size):
            
            for j in range(sequence_length):
                # Reset the gradient to prevent the error to propagate
                lstm.reset()

                # Get the target for this iteration
                y_t = data_train[i+j+1]

                # Select the data that will feed the lstm cell
                x_t = train_predictions[i+j]

                # Forward propagation
                cache = lstm.forward(x_t)

                # Compute the derivative of the loss, i.e the MSE
                dloss = 2 * (lstm.h_t - y_t)

                # Backward propagation
                lstm.backward(dloss, x_t, cache)

                # Update the weights
                lstm.update(learning_rate, lstm.optimizer)

            for s in range(predict_size):
                # Once the lstm has been fed, append the predicted values
                train_predictions.append(lstm.h_t[s])

            loss = (lstm.h_t - y_t) ** 2
            train_loss += loss

        # If wanted, computes the validation loss of the long term prediction
        if compute_validation:
            # Make predictions on the validation set
            lstm_copy = copy.deepcopy(lstm)
            for i in range(0, len(data_val) - sequence_length - predict_size, predict_size):

                y_expected = data_val[i+sequence_length]

                for j in range(sequence_length):
                    x_t = val_predictions[i+j]
                    lstm_copy.forward(x_t)

                for s in range(predict_size):
                    val_predictions.append(lstm_copy.h_t[s])

                val_loss += (lstm_copy.h_t - y_expected) ** 2

        
        if compute_validation:
            print("Epoch", epoch, "- Training loss =", train_loss.flatten() / len(data_train), "Validation loss = ",
                  val_loss.flatten() / len(data_val))
        else:
            print("Epoch", epoch, "- Training loss =", train_loss.flatten() / len(data_train))

        # Add the loss values to their respective lists for plotting
        train_loss_list.append(train_loss.flatten() / len(data_train))
        if compute_validation:
            val_loss_list.append(val_loss.flatten() / len(data_val))

    if compute_validation:
        return lstm, train_predictions, val_predictions, train_loss_list, val_loss_list
    else:
        return lstm, train_predictions, train_loss_list


def preprocess_sequential_data(file_path, sequence_length, predict_size, train_ratio=0.6, validation_ratio=0.2, power = 'p_cons'):
    # Read the dataset
    dataset = pd.read_csv(file_path, usecols=['ts', power], index_col='ts', parse_dates=['ts'])
    dataset = dataset.dropna()

    input_data = dataset[power].to_numpy()

    # Normalize the data using min-max normalization
    input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

    # Split the data into training and testing sets
    num_samples = len(input_data)
    num_training_samples = int(num_samples * train_ratio)
    num_val_samples = int(num_samples * (train_ratio + validation_ratio))

    data_train = input_data[:num_training_samples]
    data_val = input_data[num_training_samples:num_val_samples]
    data_test = input_data[num_val_samples:]

    return data_train, data_val, data_test


def write_weights_biases_to_file(lstm, file_path):
    """
    Saves the weights and biases on a txt file
    """
    with open(file_path, "w") as file:
        # Write the values of the weights
        file.write("W_gates[input]:\n")
        file.write(str(lstm.W_gates["input"]) + "\n")

        file.write("W_gates[output]:\n")
        file.write(str(lstm.W_gates["output"]) + "\n")

        file.write("W_gates[forget]:\n")
        file.write(str(lstm.W_gates["forget"]) + "\n")

        file.write("W_candidate:\n")
        file.write(str(lstm.W_candidate) + "\n")

        # Write the values of the biases
        file.write("b_gates[input]:\n")
        file.write(str(lstm.b_gates["input"]) + "\n")

        file.write("b_gates[output]:\n")
        file.write(str(lstm.b_gates["output"]) + "\n")

        file.write("b_gates[forget]:\n")
        file.write(str(lstm.b_gates["forget"]) + "\n")

        file.write("b_candidate:\n")
        file.write(str(lstm.b_candidate) + "\n")

    print("Weights and biases saved to file:", file_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--pred_size', type=int, default=1)
    parser.add_argument('--file_path', type=str, default='Data/CDB002.csv')
    parser.add_argument('--epoch', type=int, default=10)

    args = parser.parse_args()

    # The number of time data you want to use for the prediction
    sequence_length = args.seq_len
    # You also need to choose the prediction size which should be the same as the hidden size.
    predict_size = args.pred_size
    # The path of the file you want to predict from
    file_path = args.file_path
    num_epochs = args.epoch



    #Preprocess
    data_train, data_val, data_test = preprocess_sequential_data(file_path, sequence_length, predict_size)

    # Set up the LSTM
    lstm = LSTM(hidden_size=predict_size)
    
    #Set up for a non adam optimizer
    learning_rate = 0.001

    # Training
    # lstm, train_predictions, val_predictions, train_loss_list, val_loss_list = train_lstm(lstm, data_train, data_val, sequence_length, predict_size, num_epochs, learning_rate, compute_validation=True)
    lstm, predictions, train_loss_list = train_lstm(lstm, data_train, data_val, sequence_length, predict_size, num_epochs, learning_rate, compute_validation=False)


    # Path to the file wheree the trained parameters are written
    param_file_path = "Saved_parameters/long_term/weights_biases.txt"  
    write_weights_biases_to_file(lstm, param_file_path)


    plt.plot(range(num_epochs), train_loss_list, label="Training loss")
    plt.plot(range(num_epochs), val_loss_list, label="Validation loss")
    plt.legend()
    plt.show()


    plt.plot(data_train, label='Expected value')
    plt.plot(train_predictions, label='Predictions on training set')
    plt.legend()
    plt.show()

    print(val_predictions)
    plt.plot(data_val, label='Expected value')
    plt.plot(val_predictions, label='Predictions on validation set')
    plt.legend()
    plt.show()

