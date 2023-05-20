# Code Execution Guide

This guide explains how to run the Python code with command-line arguments.

Two scripts can be ran. The "train.py" trains the model while the "predict.py" makes a prediction from the trained model. 

The available arguments for train.py are:

- `--seq_len`: Length of the sequence on which the model base its prediction (i.e size of the sliding window) (default: 5)
- `--pred_size`: Number of values predicted per iteration (default: 1)
- `--file_path`: Path to the file containing the data used for training the model (default: Data/CDB002.csv)

The available arguments for predict.py are:

- `--seq_len`: Length of the sequence on which the model base its prediction (i.e size of the sliding window) (default: 5)
- `--pred_size`: Number of values predicted per iteration (default: 1)
- `--file_path`: Path to the file containing the data used for testing the model (default: Data/CDB002.csv)
- `--parameters_file_path`: Path to the txt file containing the trained weights and biases (default: Saved_parameters/weights_biases.txt).

ATTENTION : Pay attention that for a prediction, the parameters_file_path you use should have be trained with the same "seq_len" and "pred_size" you are using for the prediction.

Specify the argument values by using the format `--<argument_name>=<value>`. For example:

	python train.py --sequence_length=10 --predict_size=3 --file_path=Data/CDB002.csv
	python predict.py --sequence_length=10 --predict_size=3 --file_path=Data/CDB002.csv --parameters_file_path=weights_biases.txt


## Requirements : 

The code requires the libraries :

- `numpy==1.21.0`
- `pandas==1.3.0`
- `matplotlib==3.4.3`

These can be dowloaded from the 'requirements.txt' file.