import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import sys

class NeuralNetwork():
    def __init__(self) -> None:
        self.model = Sequential()
        self.model.add(Dense(11, activation='relu', input_shape=(11,)))
        self.model.add(Dense(1))
    
    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def load(self, model_name):
        loaded_model = tf.keras.models.load_model(model_name)
        self.model = loaded_model

    def inference(self, vector):
        vector_batch = np.expand_dims(vector, axis=0)
        original_stdout = sys.stdout
        sys.stdout = open('NUL', 'w') 
        predictions = self.model.predict(vector_batch)
        sys.stdout = original_stdout
        return predictions[0][0]
    
