
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from Code.functions import analytical_u


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

def g_trial_tf(X, model, N_output):
    x = X[:, 0:1]
    t = X[:, 1:2]

    # NN output
    N_val = model(X)

    # h1 = initial shape (satisfies IC and BC)
    h1 = (1.0 - t) * u(x)  # u(x) is initial condition function

    # h2 = correction term using NN
    h2 = x * (1.0 - x) * t * N_val

    return h1 + h2


def compute_loss(model, X_points):
    # X_points shape (N_samples, 2): column 0 is x, column 1 is t
    x = X_points[:, 0:1] # x coordinates
    t = X_points[:, 1:2] # t coordinates
    
    # We need to watch variables x and t to compute second derivatives w.r.t x
    # and first derivatives w.r.t t.
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x) # Watch x for second derivatives
        tape2.watch(t) # Watch t for first derivatives
        
        # Inner tape to compute first derivatives
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x) # Watch x for first derivatives
            tape1.watch(t) # Watch t for first derivatives
            
            # Combine inputs for the NN
            inputs = tf.stack([x[:,0], t[:,0]], axis=1) # Shape (N_samples, 2)
            N_output = model(inputs)              # Output of the network
            
            # Trial solution g_t
            X = tf.stack([x[:, 0], t[:, 0]], axis=1)
            g_t = g_trial_tf(X, model, N_output)

        
        # Calculate first order derivatives
        d_g_t_dt = tape1.gradient(g_t, t)
        d_g_t_dx = tape1.gradient(g_t, x)
    
    # Calculate second order derivatives (d^2(g_t) / dx^2)
    # Requires derivative of the output of the inner tape (d_g_t_dx) w.r.t x
    d2_g_t_d2x = tape2.gradient(d_g_t_dx, x)
    
    # PDE Residual: R = d(g_t)/dt - d^2(g_t)/dx^2 
    # The PDE is (d/dt)u = (d^2/dx^2)u, so we want R = ut - uxx = 0
    residual = d_g_t_dt - d2_g_t_d2x
    
    # Cost function (MSE of the residual)
    loss = tf.reduce_mean(tf.square(residual))
    
    return loss


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
    u_pred = g_trial_tf(X_tf, model, N_out).numpy().reshape(nt, nx)

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
    u_pred_flat = g_trial_tf(X_tf, model, N_out).numpy()

    # Reshape back to 2D grid (nt × nx)
    u_pred = u_pred_flat.reshape(nt, nx)

    return X, T, u_pred
