# Non-Dispersive Fiber Autoencoder in PyTorch

This repository contains a **PyTorch** implementation of an autoencoder-based communication system for a **non-dispersive** optical-fiber channel. It follows the approach described in:

> **Shen Li, Christian Häger, Nil Garcia, and Henk Wymeersch**  
> *“Achievable Information Rates for Nonlinear Fiber Communication via End-to-end Autoencoder Learning,”*  
> in Proc. **European Conference on Optical Communication (ECOC)**, 2018.  
> ([arXiv:1804.07675](https://arxiv.org/abs/1804.07675))

The original TensorFlow reference code can be found here:  
<https://github.com/henkwymeersch/AutoencoderFiber>

---

## Overview

- **Goal**: Learn an end-to-end physical-layer autoencoder for **non-dispersive** fiber channels with Kerr nonlinearity and additive noise.  
- **Method**: The transmitter and receiver are modeled as neural networks, jointly trained to maximize the mutual information (or equivalently, minimize cross-entropy).  
- **Non-Dispersive Channel**: This code simulates piecewise nonlinear phase shifts plus Gaussian noise at each amplification stage, *ignoring* dispersion effects.  

---

## Usage

1. **Install Dependencies**:  
   - Python 3.7+  
   - PyTorch  
   - NumPy, Matplotlib  

2. **Run the code**:  
   ```bash
   python fiber_autoencoder.py
