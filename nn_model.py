import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten

class NeuralNetwork():
    def __init__(self) -> None:
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(10,)))
        self.model.add(Dense(10, activation='relu', input_shape=(10,)))
        self.model.add(Dense(1))

    
    
