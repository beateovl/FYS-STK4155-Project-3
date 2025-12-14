
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from Code.functions import analytical_u

import tensorflow as tf

from scipy.optimize import minimize


class TF_LBFGS:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        # Store shapes AND dtypes of all trainable variables
        self.shapes = [v.shape for v in model.trainable_variables]
        self.dtypes = [v.dtype for v in model.trainable_variables]

    def pack_weights(self):
        # Flatten all weights into a single 1D tensor
        flat_vars = [tf.reshape(v, [-1]) for v in self.model.trainable_variables]
        return tf.concat(flat_vars, axis=0)

    def unpack_weights(self, flat):
        # flat is a 1D numpy array from SciPy (float64)
        idx = 0
        new_vars = []
        for shape, dtype in zip(self.shapes, self.dtypes):
            size = int(np.prod(shape))
            slice_np = flat[idx:idx + size]              # numpy slice, float64
            slice_tf = tf.reshape(slice_np, shape)       # tf.Tensor, float64
            slice_tf = tf.cast(slice_tf, dtype=dtype)    # cast to original dtype
            new_vars.append(slice_tf)
            idx += size

        # Assign back to the model
        for var, new in zip(self.model.trainable_variables, new_vars):
            var.assign(new)

    def loss_and_grad(self, flat_params, X):
        # Update model weights from flattened vector
        self.unpack_weights(flat_params)

        with tf.GradientTape() as tape:
            loss = self.loss_fn(self.model, X)

        grads = tape.gradient(loss, self.model.trainable_variables)
        # Flatten gradients to 1D numpy array (float64)
        grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        return loss.numpy().astype(np.float64), grads_flat.numpy().astype(np.float64)

    def minimize(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        x0 = self.pack_weights().numpy()  # initial flat weights (float32 -> numpy)

        result = minimize(
            fun=lambda w: self.loss_and_grad(w, X)[0],
            x0=x0,
            jac=lambda w: self.loss_and_grad(w, X)[1],
            method="L-BFGS-B",
            options={"maxiter": 500, "maxcor": 50}
        )

        # Final weights
        self.unpack_weights(result.x)
        print("L-BFGS done. Success:", result.success, "| Final loss:", result.fun)


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

def u(x):
    return tf.sin(np.pi * x)

def g_trial_tf(X, N_val):
    x = X[:, 0:1]
    t = X[:, 1:2]

    h1 = (1.0 - t) * u(x)
    h2 = x * (1.0 - x) * t * N_val
    return h1 + h2



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




def make_train_step(model, optimizer, compute_loss):
    @tf.function
    def train_step(X_points):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, X_points)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return train_step





def compute_MSE(model, nx=50, nt=50, T_final=0.5):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T_final, nt)
    X, T = np.meshgrid(x, t)

    X_flat = X.reshape(-1, 1).astype(np.float32)
    T_flat = T.reshape(-1, 1).astype(np.float32)
    X_input = np.hstack([X_flat, T_flat]).astype(np.float32)

    X_tf = tf.convert_to_tensor(X_input, dtype=tf.float32)
    N_out = model(X_tf)
    u_pred = g_trial_tf(X_tf, N_out).numpy().reshape(nt, nx)

    u_true = analytical_u(X, T)
    mse = np.mean((u_pred - u_true)**2)
    return mse


def compute_solution_grid(model, nx=50, nt=50, T_final=0.5):
    """
    Evaluate the trained PINN solution u(x,t) on a regular grid.
    
    Returns:
        X, T : 2D meshgrids
        u_pred : 2D array of shape (nt, nx)
    """
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
