import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

class MPNN(keras.Model):
    def __init__(self, hidden_layers, input_size, output_size):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.inputs_size = input_size
        self.output_size = output_size

        self.input_layer = keras.layers.InputLayer(input_shape=(input_size,))
        self.layers_set = []
        for ls, ac in hidden_layers:
            if ac is None:
                self.layers_set.append(Dense(ls))
            else:
                self.layers_set.append(Dense(ls, activation=ac))
        self.layers_set.append(Dense(output_size))
        # self.model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
        self.init_weights()

    def call(self, inputs, training=None, mask=None):
        z = self.input_layer(inputs)
        for layer in self.layers_set:
            z = layer(z)
        return z

    def loss_function(self, pred_y, y):
        return keras.backend.mean(keras.losses.mean_squared_error(y, pred_y))

    def compute_loss(self, x, y):
        logits = self.call(x)
        mse = self.loss_function(logits, y)
        return mse, logits

    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            loss, _ = self.compute_loss(x, y)
        return tape.gradient(loss, self.trainable_variables), loss

    def apply_gradients(self, optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))

    def train(self, x, y, optimizer):
        gradients, loss = self.compute_gradients(x, y)
        self.apply_gradients(optimizer, gradients, self.trainable_variables)
        return loss

    def init_weights(self):
        random_input = np.random.random(size=(1, self.inputs_size))
        self.call(tf.cast(random_input, dtype=tf.float32))

def copy_model(model):
    copied_model = MPNN(model.hidden_layers, model.inputs_size, model.output_size)
    copied_model.init_weights()
    copied_model.set_weights(model.get_weights())
    return copied_model

