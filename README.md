
				██         ██████   ██████████   █████       ██████
				██        ██            ██         ██  ██   ██  ██
				██         █████        ██         ██     ██    ██
				██            ██       ██         ██           ██
				███████  ██████        ██         ██           ██




						 PROJECT ROOT

		|-- Data/                             	# Contains all datas 
		|    |-- CDB001.py                    	# An example of file
		|                      

		|-- long_term_prediction/             	# LSTM for future prediction 
		|         |      
		|         |-- Saved_parameters/       	# Contains weights and biases obtains after training

		|-- Keras_MODEL.ipynb                 	# Keras based LSTM 

		|-- LSTM_Class.py                     	# Declaration of LSTM class

		|-- requirements.txt                  	# Packages

		|-- lstm_train.py             	       	# Contains the definition of the forward and reverse train
		|-- preprocess.py                  	# LSTM to train a short sequence
		|-- main_short_term_prediction.ipynb	# Contains an example of execution and the analysis part 

		|-- ************************************************************************

## Code Execution Guide the main work about short term prediction in time

To execute the project and observe the workflow, you can use the main.ipynb file. This Jupyter Notebook serves as an illustrative example of how to run the project and includes an analysis section where the trained model is utilized to make predictions and evaluate the results.

By opening the main.ipynb file in a Jupyter Notebook environment, you can follow along with the code and explanations provided. The notebook demonstrates the various steps involved in the project, such as data preprocessing, model training, and result analysis.

## Code Execution Guide LSTM for future prediction 
To execut the LSTM for future prediction 

This guide explains how to run the Python code with command-line arguments.

Two scripts can be ran. The "short_term_train.py" trains the model while the "short_term_predict.py" makes a prediction from the trained model. 

The available arguments for short_term_train.py are:

- `--seq_len`: Length of the sequence on which the model base its prediction (i.e number of values used for prediction) (default: 5)
- `--pred_size`: Number of values predicted in the same time (default: 1)
- `--file_path`: Path to the file containing the data used for training the model (default: Data/CDB002.csv)
- `--epoch`: Number of epochs of the training (default: 10)


The available arguments for short_term_predict.py are:

- `--seq_len`: Length of the sequence on which the model base its prediction (i.e number of values used for prediction) (default: 5)
- `--pred_size`: Number of values predicted in the same time (default: 1)
- `--file_path`: Path to the file containing the data used for testing the model (default: Data/CDB002.csv)
- `--parameters_file_path`: Path to the txt file containing the trained weights and biases (default: Saved_parameters/weights_biases.txt).

ATTENTION : Pay attention that for a prediction, the parameters_file_path you use should have be trained with the same "seq_len" and "pred_size" you are using for the prediction.

Specify the argument values by using the format `--<argument_name>=<value>`. For example:

	python short_term_train.py --sequence_length=10 --predict_size=3 --file_path=Data/CDB002.csv
	python short_term_predict.py --sequence_length=10 --predict_size=3 --file_path=Data/CDB002.csv --parameters_file_path=weights_biases.txt

Or, if you want to run the code with all default parameters, you can simply do :

	python short_term_train.py 
	python short_term_predict.py


### Requirements : 

The code requires the libraries :

- `numpy==1.21.0`
- `pandas==1.3.0`
- `matplotlib==3.4.3`

These can be dowloaded from the 'requirements.txt' file.
