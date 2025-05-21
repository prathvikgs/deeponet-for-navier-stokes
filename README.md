# deeponet-for-navier-stokes
# 🌀 DeepONet Solver for 2D Incompressible Navier-Stokes Equations

This repository implements a **Deep Operator Network (DeepONet)** to solve the **2D incompressible Navier-Stokes equations** using physics-informed learning and a Spectral Projection Inspired (SPI) architecture. The model approximates the mapping from a parametric forcing function to the corresponding velocity and pressure fields over time.

---

## 📘 Table of Contents

- [Problem Description](#📖-problem-description)
- [DeepONet Architecture](#🧠-deeponet-architecture)
- [Features](#🔍-features)
- [Requirements](#⚙️-requirements)
- [Usage](#🚀-usage)
- [Configuration Parameters](#⚙️-configuration-parameters)
- [Output](#📤-output)
- [Extending the Code](#🧪-extending-the-code)
- [License](#📜-license)

---

## 📖 Problem Description

We solve the 2D incompressible Navier-Stokes equations:

\[
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} + \nabla p = \nu \Delta \mathbf{u} + \mu \mathbf{u} + \mathbf{f}
\]
\[
\nabla \cdot \mathbf{u} = 0
\]

Where:
- \( \mathbf{u} = (u, v) \): velocity components
- \( p \): pressure field
- \( \nu \): viscosity coefficient
- \( \mu \): damping coefficient
- \( \mathbf{f}(y) = -\frac{1}{4} \sin(ky) \): forcing function

The solution is defined over a periodic square domain \( [0, 2\pi] \times [0, 2\pi] \) for \( t \in [0, 10] \).

---

## 🧠 DeepONet Architecture

The DeepONet consists of:
- **Branch Network**: Encodes the forcing function values sampled at sensor points.
- **Trunk Network**: Encodes the spatial-temporal coordinates \( (x, y, t) \).
- **Spectral Projection Inspired Design**: The model splits both networks into two halves for \( \psi \) (stream function) and \( p \) (pressure), combining them via a custom dot product.

The output is:
- \( \psi(x, y, t) \) — stream function
- \( p(x, y, t) \) — pressure

Velocity is derived as:
- \( u = \frac{\partial \psi}{\partial y} \)
- \( v = -\frac{\partial \psi}{\partial x} \)

---

## 🔍 Features

- Physics-informed loss: Navier-Stokes PDE residuals
- Periodic boundary condition enforcement
- Initial condition training
- Gradient-based optimization using Adam with exponential decay
- TensorFlow 2.x implementation
- Option to include supervised solution data (commented out)

---

## ⚙️ Requirements

Install the required packages:

```bash
pip install tensorflow numpy scipy matplotlib pandas
