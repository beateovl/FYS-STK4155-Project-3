
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from Code.functions import analytical_u
import seaborn as sns 
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Neural Network model, using keras 
def create_network_model(input_dim=2, output_dim=1, layers=[6], activation='tanh'):
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(Dense(layers[0], activation=activation))

    for units in layers[1:]:
        model.add(Dense(units, activation=activation))
    # Output layer N(x, t, P)
    model.add(Dense(output_dim, activation='linear')) # Linear output for N
    return model


# Trial and loss functions

def u(x):
    return tf.sin(np.pi * x)

def g_trial_tf(X, N_val):
    x = X[:, 0:1]
    t = X[:, 1:2]

    h1 = (1.0 - t) * u(x)
    h2 = x * (1.0 - x) * t * N_val
    return h1 + h2


# Loss function for the PINN
def compute_loss(model, X):
    x = X[:, 0:1]
    t = X[:, 1:2]

    # First tape computes g, g_t, g_x
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])

        inputs = tf.concat([x, t], axis=1)
        N_output = model(inputs)
        g = g_trial_tf(inputs, N_output)
        
        g_t = tape.gradient(g, t)
        g_x = tape.gradient(g, x)

    # Second derivative
    g_xx = tape.gradient(g_x, x)

    del tape  # free memory

    residual = g_t - g_xx
    return tf.reduce_mean(tf.square(residual))



# Training step function
def make_train_step(model, optimizer, compute_loss):
    @tf.function
    def train_step(X_points):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, X_points)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return train_step




# Function to compute MSE against analytical solution
def compute_MSE(model, nx=50, nt=50, T_final=0.5):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T_final, nt)
    X, T = np.meshgrid(x, t)

# Flatten for NN input
    X_flat = X.reshape(-1, 1).astype(np.float32)
    T_flat = T.reshape(-1, 1).astype(np.float32)
    X_input = np.hstack([X_flat, T_flat]).astype(np.float32)

# Tensor for TF model
    X_tf = tf.convert_to_tensor(X_input, dtype=tf.float32)
    N_out = model(X_tf)
    u_pred = g_trial_tf(X_tf, N_out).numpy().reshape(nt, nx)

    u_true = analytical_u(X, T)
    mse = np.mean((u_pred - u_true)**2)
    return mse

# Function to compute PINN solution on a grid
def compute_solution_grid(model, nx=50, nt=50, T_final=0.5):
  
    # Create grid in x and t
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T_final, nt)
    X, T = np.meshgrid(x, t)

    # Flatten for NN input
    X_flat = X.reshape(-1, 1).astype(np.float32)
    T_flat = T.reshape(-1, 1).astype(np.float32)
    X_input = np.hstack([X_flat, T_flat]).astype(np.float32)

    # Tensor for TF model
    X_tf = tf.convert_to_tensor(X_input, dtype=tf.float32)

    # Network output and final trial solution
    N_out = model(X_tf)
    u_pred_flat = g_trial_tf(X_tf, N_out).numpy()


    # Reshape back to 2D grid (nt × nx)
    u_pred = u_pred_flat.reshape(nt, nx)

    return X, T, u_pred
