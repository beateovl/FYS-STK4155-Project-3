import numpy as np

# analytical

def analytical_u(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

#euler


def euler(dx, dt, T):
    L = 1.0
    Nx = int(L/dx)
    x = np.linspace(0, L, Nx+1)

    Nt = int(T/dt)
    t = np.linspace(0, T, Nt+1)

    u = np.sin(np.pi * x)      # initial condition u(x,0)
    u_new = u.copy()

    alpha = dt/dx**2

    # store snapshots if you want
    sol = np.zeros((Nt+1, Nx+1))
    sol[0,:] = u

    for n in range(Nt):
        # boundary conditions
        u[0]  = 0.0
        u[-1] = 0.0

        for i in range(1, Nx):
            u_new[i] = u[i] + alpha*(u[i+1] - 2*u[i] + u[i-1])

        u[:] = u_new
        sol[n+1,:] = u

    return x, t, sol
