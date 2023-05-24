# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
import time

def preprocess_data(file_path, sequence_length, predict_size, train_ratio=0.8, power='p_cons'):
    # Read the dataset
    print('start preprocess')
    start_time = time.time()
    dataset = pd.read_csv(file_path, usecols=['ts', power], index_col='ts', parse_dates=['ts'])
    dataset = dataset.dropna()

    # Add p_cons data to input_data
    input_data = []
    target_data = []

    # Convert dataset to numpy array
    dataset_array = dataset[power].to_numpy()

    # Generate input sequences
    input_data = np.array([dataset_array[i:i + sequence_length] for i in range(0, len(dataset_array) - sequence_length - predict_size + 1, sequence_length + predict_size)])
    # Generate target sequences
    target_data = np.array([dataset_array[i + sequence_length:i + sequence_length + predict_size] for i in range(0, len(dataset_array) - sequence_length - predict_size + 1, sequence_length + predict_size)])

    input_data = np.array(input_data)
    target_data = np.array(target_data)

    # Normalize the data using min-max normalization
    input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    target_data = (target_data - np.min(target_data)) / (np.max(target_data) - np.min(target_data))

    # Reshape input_data and target_data
    input_data = input_data.reshape(input_data.shape[0], sequence_length, 1)
    target_data = target_data.reshape(target_data.shape[0], predict_size, 1)

    # Split the data into training, validation, and testing sets
    num_samples = len(input_data)
    num_training_samples = int(num_samples * train_ratio)

    input_train = input_data[:num_training_samples]
    target_train = target_data[:num_training_samples]
    input_val = input_data[num_training_samples:]
    target_val = target_data[num_training_samples:]

    end_time = time.time()
    print("Preprocessing completed in", round(end_time - start_time, 2), "seconds")
    print("end of preprocess")

    return input_train, target_train, input_val, target_val
