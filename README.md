# deeponet-for-navier-stokes
# 🌀 DeepONet Solver for 2D Incompressible Navier-Stokes Equations

This repository implements a **Deep Operator Network (DeepONet)** to solve the **2D Navier-Stokes equations** using a physics-informed deep operator network with modifications. The model approximates the mapping from a parametric forcing function to the corresponding velocity and pressure fields over time.

---
∂u/∂t + u·∇u + ∇p = νΔu + μu + f


Where:
- `u = (u, v)` is the velocity field (2D)
- `p` is pressure
- `ν` is the viscosity coefficient
- `μ` is the damping coefficient
- `f(y) = -(1/4) * sin(k * y)` is the forcing function

The domain is a periodic square: `[0, 2π] x [0, 2π]`, with time `t` in `[0, 10]`.

---

## DeepONet Architecture

The DeepONet consists of:
- **Branch Network**: Encodes the input forcing function sampled at `sensors` points.
- **Trunk Network**: Encodes spatial-temporal coordinates `(x, y, t)`.

The network is split to model:
- `ψ(x, y, t)` — stream function
- `p(x, y, t)` — pressure

Velocity is computed from `ψ` as:
- `u = ∂ψ/∂y`
- `v = -∂ψ/∂x`

---

## Features

- Physics-informed loss using Navier-Stokes PDE residuals
- Periodic boundary condition enforcement
- Initial condition loss minimization
- Learning from both labeled and unlabeled data
- Gradient-based optimization using Adam
- Supports optional supervised training (commented out)

---

## Requirements

Install dependencies with


pip install tensorflow numpy scipy matplotlib pandas

Usage
1. Run the script
python ns_deeponet_final.py
This will:
Build and train the DeepONet model
Save the trained model as deeponet_Navierstokes.keras


Inside ns_deeponet_final.py, you can adjust the following:
sensors = 500             # Number of sensor points
spacial_resol = 128       # Grid resolution (x and y)
temporal_resol = 50       # Number of time steps
total_epochs = 2000       # Training epochs
mu = 0.1                  # Damping coefficient
nu = 0.001                # Viscosity
rho = 1                   # Fluid density
k = np.array([4])         # Wavenumber for the forcing function

The trained model is saved as: deeponet_Navierstokes.keras
Loss histories: deeponet.loss_eqn, deeponet.loss_b, deeponet.loss_in, deeponet.loss_total
To evaluate the solution at new inputs, use:
p, u, v, *_ = deeponet.solution(X_branch, X_trunk)


