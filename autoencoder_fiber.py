import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

###############################################################################
# Basic Parameters
###############################################################################
P_in_dBm = -2          # Input power in dBm
gamma = 1.27           # Fiber nonlinearity factor
M = 16                 # Constellation size
tx_layers = 2          # Number of layers in the transmitter
rx_layers = 3          # Number of layers in the receiver
neurons_per_layer = 50 # Hidden dimension
learning_rate = 1e-3   # Adam learning rate
iterations = 50000      # Training iterations
stacks = 20            # Repeat each symbol 'stacks' times
channel_uses = 2       # Must be 2 (I/Q)
L = 2000               # Fiber length
K = 20                 # Number of amplification segments
sigma = 3.8505e-4 * np.sqrt(2)  # Noise scale
P_in = 10**(P_in_dBm/10) * 0.001 # Convert dBm to linear scale (Watts)
Ess = np.sqrt(P_in)    # sqrt of input power


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# Model/Channel Definitions
###############################################################################
def normalization(x, Ess):
    """
    Enforce average power E[|x|^2] = Ess^2 for 2D signals x.
    x has shape (B, 2).
    """
    power = 2.0 * torch.mean(x**2)  # factor 2 for real+imag
    factor = Ess / torch.sqrt(power)
    return x * factor

def fiber_channel(x):
    """
    Non-dispersive fiber channel with nonlinear phase + noise in K segments.
    x shape: (B, 2).
    Returns shape (B, 3): [real, imag, power].
    """
    xr, xi = x[:, 0], x[:, 1]
    for _ in range(K):
        s = gamma * (xr**2 + xi**2) * (L / K)
        old_xr = xr
        xr = xr * torch.cos(s) - xi * torch.sin(s)
        xi = old_xr * torch.sin(s) + xi * torch.cos(s)
        # Noise
        xr += sigma * torch.randn_like(xr)
        xi += sigma * torch.randn_like(xi)
    power = xr**2 + xi**2
    return torch.stack([xr, xi, power], dim=1)

def build_sequential(in_dim, out_dim, hidden_dim, num_layers):
    layers = []
    curr_in = in_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(curr_in, hidden_dim))
        layers.append(nn.Tanh())
        curr_in = hidden_dim
    # Final layer: raw logits, no softmax
    layers.append(nn.Linear(curr_in, out_dim))
    return nn.Sequential(*layers)

###############################################################################
# Plotting
###############################################################################

def plot_results(
    tx_net, 
    rx_net, 
    labels, 
    loss_history, 
    normalization_fn, 
    device, 
    Ess, 
    M
):
    # 1) Plot training loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 2) Plot the learned constellation
    tx_net.eval()
    # Convert integer labels to one-hot
    x_one_hot = torch.nn.functional.one_hot(labels, num_classes=M).float().to(device)

    with torch.no_grad():
        x_enc = tx_net(x_one_hot)           # (stacks*M, 2)
        x_enc_norm = normalization_fn(x_enc, Ess)
        y_out = fiber_channel(x_enc_norm) 
    x_enc_np = x_enc_norm.cpu().numpy()
    x_const = x_enc_np[:, 0]
    y_const = x_enc_np[:, 1]

    xmax = np.max(np.abs(x_const))
    ymax = np.max(np.abs(y_const))
    max_axis = 1.2 * max(xmax, ymax)

    plt.figure(figsize=(5,5))
    plt.scatter(x_const, y_const, c='b', marker='o')
    plt.axis('equal')
    plt.xlim(-max_axis, max_axis)
    plt.ylim(-max_axis, max_axis)
    plt.title("Learned Constellation (Normalized)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.show()

    # 3) Decision Regions
    rx_net.eval()

    resolution = 500
    xv = np.linspace(-max_axis, max_axis, resolution)
    yv = np.linspace(-max_axis, max_axis, resolution)
    xx, yy = np.meshgrid(xv, yv)
    x_flat = xx.ravel()  # (resolution^2,)
    y_flat = yy.ravel()
    r_flat = x_flat**2 + y_flat**2
    xy_inputs = np.stack([x_flat, y_flat, r_flat], axis=1)  # (resolution^2, 3)

    # Pass through rx_net => raw logits => argmax
    G_torch = torch.from_numpy(xy_inputs).float().to(device)
    with torch.no_grad():
        logits_grid = rx_net(G_torch)
    decisions = logits_grid.argmax(dim=1).cpu().numpy()  # (resolution^2,)

    z = decisions.reshape(resolution, resolution)
    
    # constellation after the channel:
    y_np = y_out.cpu().numpy()
    x_rx = y_np[:, 0]
    y_rx = y_np[:, 1]

    plt.figure(figsize=(6,6))
    plt.title("Decision Regions")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.pcolormesh(xx, yy, z, shading='auto', cmap='rainbow')
    plt.scatter(x_rx, y_rx, c='black', marker='o', s=30, label='Constellation')
    plt.legend()
    plt.show()

###############################################################################
# 4) Main function
###############################################################################
def main():

    # Prepare integer labels
    labels_np = np.arange(M)                 # shape (M,)
    labels_np = np.tile(labels_np, stacks)   # shape (stacks*M,) repeated
    labels = torch.from_numpy(labels_np).long().to(device)  # integer labels

    # Define a function to produce one-hot vectors *on the fly*:
    def to_one_hot(label_vec, num_classes):
        # label_vec: shape (B, ) of ints
        # returns shape (B, num_classes) of 0/1
        return torch.nn.functional.one_hot(label_vec, num_classes=num_classes).float()


    # Build transmitter/receiver
    # The transmitter expects shape: (B, M) for input,
    tx_net = build_sequential(
        in_dim=M, 
        out_dim=channel_uses, 
        hidden_dim=neurons_per_layer, 
        num_layers=tx_layers
    ).to(device)

    rx_net = build_sequential(
        in_dim=channel_uses + 1, 
        out_dim=M, 
        hidden_dim=neurons_per_layer, 
        num_layers=rx_layers
    ).to(device)

    optimizer = optim.Adam(
        list(tx_net.parameters()) + list(rx_net.parameters()), 
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()  
    # CrossEntropyLoss expects: 
    #   logits shape (B, M)
    #   labels shape (B,) with integer classes 0..M-1

    # Training Loop
    loss_history = []
    for i in range(iterations):
        optimizer.zero_grad()
        x_one_hot = to_one_hot(labels, M)  # shape: (stacks*M, M)
        x_enc = tx_net(x_one_hot)           # shape: (stacks*M, 2)
        x_enc_norm = normalization(x_enc, Ess)
        y_out = fiber_channel(x_enc_norm)   # shape: (stacks*M, 3)
        logits = rx_net(y_out)             # shape: (stacks*M, M)
        loss_val = criterion(logits, labels)

        loss_val.backward()
        optimizer.step()

        loss_history.append(loss_val.item())

        if (i+1) % (iterations/10) == 0 or i == 1:
            MI_bits = (np.log(M) - loss_val.item() * (1.0)) / np.log(2.0)
            print(f"Iteration {i+1}, Loss: {loss_val.item():.5f}, MI: {MI_bits:.5f} bits")

    plot_results(
        tx_net=tx_net,
        rx_net=rx_net,
        labels=labels,
        loss_history=loss_history,
        normalization_fn=normalization,
        device=device,
        Ess=Ess,
        M=M
    )

if __name__ == "__main__":
    main()
