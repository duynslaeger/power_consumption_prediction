from LSTM_Class import *
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

def train_lstm(lstm, data_train, data_test, sequence_length, predict_size, num_epochs, learning_rate, compute_validation=False):
    train_loss_list = []
    val_loss_list = []

    predictions = data_train[:sequence_length]

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        for i in range(len(data_train) - sequence_length - predict_size):
            # Get the input and target for this iteration
            y_t = data_train[i]
            lstm.reset()
            for j in range(sequence_length):
                
                x_t = predictions[i+j]

                # Forward pass
                cache = lstm.forward(x_t)

                # Compute the loss and its gradient MSE
                dloss = 2 * (lstm.h_t - y_t)

                # Backward pass
                lstm.backward(dloss, x_t, cache)

                # Update the weights
                lstm.update(learning_rate, lstm.optimizer)

            #TO DO : append le résultat dans le predictions
            loss = (lstm.h_t - y_t) ** 2
            train_loss += loss

        if compute_validation:
            # Make predictions on the test set
            lstm_copy = copy.deepcopy(lstm)
            for i in range(len(data_train) - sequence_length - predict_size):
                y_expected = data_train[i+sequence_length]
                for j in range(sequence_length):
                    x_t = data_train[i+j]
                    lstm_copy.forward(x_t)

                val_loss += (lstm_copy.h_t - y_expected) ** 2

        if epoch % 10 == 0:
            if perform_predictions:
                print("Epoch", epoch, "training loss", train_loss.flatten() / len(data_train), "Validation loss",
                      val_loss.flatten() / len(input_test))
            else:
                print("Epoch", epoch, "training loss", train_loss.flatten() / len(data_train))

        # Add the loss values to their respective lists for plotting
        train_loss_list.append(train_loss.flatten() / len(data_train))
        if perform_predictions:
            val_loss_list.append(val_loss.flatten() / len(input_test))

    if compute_validation:
        return lstm, train_loss_list, val_loss_list
    else:
        return lstm, train_loss_list


def preprocess_sequential_data(file_path, sequence_length, predict_size, train_ratio=0.8, power = 'p_cons'):
    # Read the dataset
    dataset = pd.read_csv(file_path, usecols=['ts', power], index_col='ts', parse_dates=['ts'])
    dataset = dataset.dropna()

    input_data = copy.copy(dataset[power])

    # Normalize the data using min-max normalization
    input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

    # Split the data into training and testing sets
    num_samples = len(input_data)
    num_training_samples = int(num_samples * train_ratio)

    data_train = input_data[:num_training_samples]
    data_test = input_data[num_training_samples:]

    return data_train, data_test



#The number of time data you want to use for the prediction
sequence_length = 1

#You also need to choose the prediction size which should be the same as the hidden size.
predict_size = 3

#Preprocess
data_train, data_test = preprocess_sequential_data('CDB002.csv', sequence_length, predict_size)

# Set up the LSTM
lstm = LSTM(hidden_size=predict_size)

# Train the LSTM
num_epochs = 30
#Set up for a non adam optimizer
learning_rate = 0.001

#training
lstm, train_loss_list, val_loss_list = train_lstm(lstm, data_train, data_test, sequence_length, predict_size, num_epochs, learning_rate, perform_predictions=True)