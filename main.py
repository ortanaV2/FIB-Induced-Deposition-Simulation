import numpy as np
import matplotlib.pyplot as plt

Nx, Nz = 300, 600
dr, dz = 0.01, 0.01
r_max, z_max = Nx*dr, Nz*dz
V0 = 30_000

linsen = {
    "Linse1": {"z": 1.0, "depth": 0.25, "r_in": 0.1, "r_out": 1.0, "V": V0, "I": 5e-3},
    "Linse2": {"z": 2.0, "depth": 0.1,  "r_in": 0.15, "r_out": 1.25, "V": V0, "I": 5e-3},
    "Linse3:": {"z": 2.5, "depth": 0.3,  "r_in": 0.15, "r_out": 1, "V": 6000, "I": 5e-3}
}

phi = np.zeros((Nx, Nz))

# Electrodes in the potential field
for linse in linsen.values():
    z_start = int(linse["z"] / dz)
    z_end = int((linse["z"] + linse["depth"]) / dz)
    r_in = int(linse["r_in"] / dr)
    r_out = int(linse["r_out"] / dr)

    phi[Nx//2 + r_in:Nx//2 + r_out, z_start:z_end] = linse["V"]
    phi[Nx//2 - r_out:Nx//2 - r_in, z_start:z_end] = linse["V"]

# Laplace-Iteration with Jacobi-Method
phi_new = phi.copy()
for it in range(5000):
    phi_new[1:-1, 1:-1] = 0.25 * (
        phi[2:, 1:-1] + phi[:-2, 1:-1] +
        phi[1:-1, 2:] + phi[1:-1, :-2]
    )
    for linse in linsen.values():
        z_start = int(linse["z"] / dz)
        z_end = int((linse["z"] + linse["depth"]) / dz)
        r_in = int(linse["r_in"] / dr)
        r_out = int(linse["r_out"] / dr)
        phi_new[Nx//2 + r_in:Nx//2 + r_out, z_start:z_end] = linse["V"]
        phi_new[Nx//2 - r_out:Nx//2 - r_in, z_start:z_end] = linse["V"]
    # Update
    phi, phi_new = phi_new, phi

# Calculating electric field components
Er = -(phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2*dr)
Ez = -(phi[1:-1, 2:] - phi[1:-1, :-2]) / (2*dz)
E = np.sqrt(Er**2 + Ez**2)

fig, ax = plt.subplots(figsize=(10, 6))
extent = [0, z_max, -r_max/2, r_max/2]
im = ax.imshow(E, cmap='plasma', origin='lower', extent=extent, aspect='auto')
cb = plt.colorbar(im, ax=ax, label='|E| (V/cm)')

ax.set_title("Electrostatic Field Strength")
ax.set_xlabel("z (cm)")
ax.set_ylabel("r (cm)")
plt.tight_layout()
plt.show()
