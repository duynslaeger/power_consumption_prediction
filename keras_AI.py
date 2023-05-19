# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

def scale(dataset, _min, _max):
    new_dataset = ((dataset - _min) / (_max - _min))
    return new_dataset

def unscale(dataset, _min, _max):
    new_dataset = dataset * (_max - _min) + _min
    return new_dataset

def outlier(df):

    z_score = stats.zscore(df)
    filtered = (np.abs(z_score) < 2).all(axis = 1)
    new_df = df[filtered]

    return new_df
# from datetime import datetime

# Read the csv file

dataset = pd.read_csv('CDB005_15min.csv', usecols=['ts', 'p_cons'], index_col='ts', parse_dates=['ts'])
#dataset = outlier(dataset)

_min = dataset.min()
_max = dataset.max()
dataset_scaled = scale(dataset, _min, _max)
size = int(len(dataset) * 0.95)
seq_len = 1

trainGenerator = TimeseriesGenerator(dataset_scaled[:size].to_numpy(), dataset_scaled[:size].to_numpy(), length=seq_len)
testGenerator = TimeseriesGenerator(dataset_scaled[size:].to_numpy(), dataset_scaled[size:].to_numpy(), length=seq_len)

model = Sequential()
model.add(InputLayer(input_shape=(seq_len, 1)))
model.add(LSTM(units=3000))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mae')


# fit the model
model.fit(trainGenerator, epochs=10, batch_size=128)

prediction = model.predict(testGenerator)

fig = plt.figure(figsize=(15, 7), dpi=100)
plt.plot(prediction, lw=0.7, label='prediction')
'''
Comme la taille de la séquence est 11, la première valeur prédite par le modèle est la 12-ième, il faut donc enlever les 11 premières valeurs du jeu de test.
Ce phénomène est propre à l'entrainement des modèles de Keras avec un TimeseriesGenerator
'''
plt.plot(dataset_scaled[size:].to_numpy()[1:], lw=0.7, label='real')
plt.legend()
