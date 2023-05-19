import LSTM_Class
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt



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