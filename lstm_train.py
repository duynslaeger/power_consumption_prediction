# -*- coding: utf-8 -*-
from LSTM_Class import *

def train_lstm_reverse(lstm, input_train, target_train, input_val, target_val, num_epochs, learning_rate, perform_predictions=True):
    print(' ')
    print(' ')
    start_time = time.time()
    print('start of training with train lstm forward')
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        for i in range(len(input_train)):
            # Get the input and target for this iteration
            y_t = target_train[i]

            cache = []
            dloss = []
            for j in range(sequence_length):
                lstm.reset()
                x_t = input_train[i][j]

                # Forward pass
                cache.append(lstm.forward(x_t))
                # Compute the loss and its gradient MSE
                dloss.append(2 * (lstm.h_t - y_t))

            # Retro backward propagation
            for j in reversed(range(sequence_length)):
                lstm.reset()
                x_t = input_train[i][j]
                lstm.backward(dloss[j], x_t, cache[j])

                # Update the weights
                lstm.update(learning_rate, lstm.optimizer)

            loss = (lstm.h_t - y_t) ** 2
            train_loss += loss

        if perform_predictions:
            # Make predictions on the test set
            lstm_copy = copy.deepcopy(lstm)
            for i in range(len(input_val)):
                y_pred = target_val[i]
                for j in range(sequence_length):
                    x_t = input_val[i][j]
                    lstm_copy.forward(x_t)

                val_loss += (lstm_copy.h_t - y_pred) ** 2

        if epoch % 10 == 0:
            if perform_predictions:
                print("Epoch", epoch, "training loss", train_loss.flatten() / len(input_train), "Validation loss",
                      val_loss.flatten() / len(input_val))
            else:
                print("Epoch", epoch, "training loss", train_loss.flatten() / len(input_train))

        # Add the loss values to their respective lists for plotting
        train_loss_list.append(train_loss.flatten() / len(input_train))
        if perform_predictions:
            val_loss_list.append(val_loss.flatten() / len(input_val))


    end_time = time.time()

    print("Training completed in", round(end_time - start_time, 2), "seconds")
    print("end of training")

    if perform_predictions:
        return lstm, train_loss_list, val_loss_list
    else:
        return lstm, train_loss_list

def train_lstm_forward(lstm, input_train, target_train, input_test, target_test, sequence_length, num_epochs, learning_rate, perform_predictions=True):
    print(' ')
    print(' ')
    start_time = time.time()
    print('start of training with train lstm forward')
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        for i in range(len(input_train)):
            # Get the input and target for this iteration
            y_t = target_train[i]
            lstm.reset()
            for j in range(sequence_length):
                
                x_t = input_train[i][j]

                # Forward pass
                cache = lstm.forward(x_t)

                # Compute the loss and its gradient MSE
                dloss = 2 * (lstm.h_t - y_t)

                # Backward pass
                lstm.backward(dloss, x_t, cache)

                # Update the weights
                lstm.update(learning_rate, lstm.optimizer)

            loss = (lstm.h_t - y_t) ** 2
            train_loss += loss

        if perform_predictions:
            # Make predictions on the test set
            lstm_copy = copy.deepcopy(lstm)
            for i in range(len(input_test)):
                y_pred = target_test[i]
                for j in range(sequence_length):
                    x_t = input_test[i][j]
                    lstm_copy.forward(x_t)

                val_loss += (lstm_copy.h_t - y_pred) ** 2

        if epoch % 10 == 0:
            if perform_predictions:
                print("Epoch", epoch, "training loss", train_loss.flatten() / len(input_train), "Validation loss",
                      val_loss.flatten() / len(input_test))
            else:
                print("Epoch", epoch, "training loss", train_loss.flatten() / len(input_train))

        # Add the loss values to their respective lists for plotting
        train_loss_list.append(train_loss.flatten() / len(input_train))
        if perform_predictions:
            val_loss_list.append(val_loss.flatten() / len(input_test))

    end_time = time.time()
    
    print("Training completed in", round(end_time - start_time, 2), "seconds")
    print("end of training")

    if perform_predictions:
        return lstm, train_loss_list, val_loss_list
    else:
        return lstm, train_loss_list
