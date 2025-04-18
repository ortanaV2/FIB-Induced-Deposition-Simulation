import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Define grid size and resolution
Nx, Nz = 300, 600
dr, dz = 0.01, 0.01  # in cm
r_max, z_max = Nx * dr, Nz * dz

lenses = {
    "Extractor": {"z": 0.5, "depth": 0.25, "r_in": 0.15, "r_out": 1.0, "V": -30000},
    "Focus1": {"z": 1, "depth": 0.15, "r_in": 0.3, "r_out": 1.25, "V": 0},
    "Accelerate1": {"z": 1.4, "depth": 0.25, "r_in": 0.25, "r_out": 1.0, "V": -30000},
    "Focus2": {"z": 1.8, "depth": 0.15, "r_in": 0.3, "r_out": 1.25, "V": 0},
    "Accelerate2": {"z": 2.15, "depth": 0.25, "r_in": 0.25, "r_out": 1.0, "V": -30000},
    "Accelerate3": {"z": 4, "depth": 0.25, "r_in": 0.25, "r_out": 1.0, "V": -60000},
}

# Initialize potential array
phi = np.zeros((Nx, Nz))
for linse in lenses.values():
    # Map lens parameters to grid indices
    z0 = int(linse["z"] / dz)
    z1 = int((linse["z"] + linse["depth"]) / dz)
    ri, ro = int(linse["r_in"] / dr), int(linse["r_out"] / dr)
    # Set potential values for the lens regions
    phi[Nx // 2 + ri:Nx // 2 + ro, z0:z1] = linse["V"]
    phi[Nx // 2 - ro:Nx // 2 - ri, z0:z1] = linse["V"]

# Laplace solver (Jacobi iteration)
phi_new = phi.copy()
for iteration in range(3000):
    # Update potential using Jacobi method
    phi_new[1:-1, 1:-1] = 0.25 * (phi[2:, 1:-1] + phi[:-2, 1:-1] +
                                  phi[1:-1, 2:] + phi[1:-1, :-2])
    # Reapply lens boundary conditions
    for linse in lenses.values():
        z0 = int(linse["z"] / dz)
        z1 = int((linse["z"] + linse["depth"]) / dz)
        ri, ro = int(linse["r_in"] / dr), int(linse["r_out"] / dr)
        phi_new[Nx // 2 + ri:Nx // 2 + ro, z0:z1] = linse["V"]
        phi_new[Nx // 2 - ro:Nx // 2 - ri, z0:z1] = linse["V"]
    # Swap arrays for the next iteration
    phi, phi_new = phi_new, phi

    if iteration % 500 == 0:
        print(f"Laplace solver iteration {iteration}/3000")

print("Laplace solver completed.")

# Calculate electric field from potential
Er = -(phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dr)  # Radial component
Ez = -(phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dz)  # Axial component
r_vals = (np.arange(1, Nx - 1) - Nx // 2) * dr
z_vals = np.arange(1, Nz - 1) * dz

@njit
def bilinear_interpolate(r_vals, z_vals, field, r, z):
    # Compute bilinear interpolation for field values at (r, z)
    dr = r_vals[1] - r_vals[0]
    dz = z_vals[1] - z_vals[0]
    i = int((r - r_vals[0]) / dr)
    j = int((z - z_vals[0]) / dz)

    # Return 0 if outside the grid
    if i < 0 or i >= field.shape[0] - 1 or j < 0 or j >= field.shape[1] - 1:
        return 0.0

    # Interpolation weights
    r1 = r_vals[i]
    z1 = z_vals[j]
    fr1z1 = field[i, j]
    fr2z1 = field[i + 1, j]
    fr1z2 = field[i, j + 1]
    fr2z2 = field[i + 1, j + 1]

    t = (r - r1) / dr
    u = (z - z1) / dz

    # Compute interpolated value
    return (1 - t) * (1 - u) * fr1z1 + t * (1 - u) * fr2z1 + (1 - t) * u * fr1z2 + t * u * fr2z2

# Simulate ion trajectories
@njit(parallel=True)
def simulate_ions(n_ionen, steps, dt, r_vals, z_vals, Er, Ez, q, m, v0, r0_max):
    # Initialize result arrays
    results_r = np.zeros((n_ionen, steps))
    results_z = np.zeros((n_ionen, steps))
    results_vr = np.zeros((n_ionen, steps))
    results_vz = np.zeros((n_ionen, steps))

    for i in prange(n_ionen):
        # Initialize ion position and velocity
        angle = np.deg2rad(np.random.uniform(-90, 90))
        r = np.random.uniform(-r0_max, r0_max)
        z = 0.0
        vr = v0 * np.sin(angle)
        vz = v0 * np.cos(angle)

        for t in range(steps):
            # Store current position and velocity
            results_r[i, t] = r
            results_z[i, t] = z
            results_vr[i, t] = vr
            results_vz[i, t] = vz

            # Interpolate electric field at current position
            Er_loc = bilinear_interpolate(r_vals, z_vals, Er, r, z)
            Ez_loc = bilinear_interpolate(r_vals, z_vals, Ez, r, z)

            # Compute acceleration from electric field
            ar = (q / m) * Er_loc * 1e2  # Radial acceleration
            az = (q / m) * Ez_loc * 1e2  # Axial acceleration

            # Update velocity and position
            vr += ar * dt
            vz += az * dt
            r += vr * dt
            z += vz * dt

            # Stop simulation if ion leaves the simulation area
            if z >= z_vals[-1] or r < r_vals[0] or r > r_vals[-1]:
                break

        if i % 5 == 0:
            print(f"Ion {i + 1}/{n_ionen} simulated")

    return results_r, results_z, results_vr, results_vz

# Constants and parameters
q, m = 1.602e-19, 40 * 1.67e-27  # Charge and mass of ion
n_ionen = 20  # Number of ions
v0 = 1e5  # Initial velocity (cm/s)
r0_max = 0.05  # Maximum initial radial position (cm)
steps = 6000  # Number of time steps
dt = 1e-9  # Time step size (1 ns)

print("Starting ion simulation...")
results_r, results_z, results_vr, results_vz = simulate_ions(
    n_ionen, steps, dt, r_vals, z_vals, Er, Ez, q, m, v0, r0_max
)
print("Ion simulation completed.")
print("Building plot...")

fig, ax = plt.subplots(figsize=(10, 6))
extent = [0, z_max, -r_max / 2, r_max / 2]
E_mag = np.sqrt(Er**2 + Ez**2)  # Magnitude of electric field
im = ax.imshow(E_mag, cmap='plasma', origin='lower', extent=extent, aspect='auto')

for i in range(n_ionen):
    r = results_r[i]
    z = results_z[i]
    vr = results_vr[i]
    vz = results_vz[i]
    speed = np.sqrt(vr**2 + vz**2)
    norm_speed = (speed - speed.min()) / (speed.max() - speed.min() + 1e-9)
    colors = plt.cm.twilight_shifted(norm_speed)

    for j in range(steps - 1):
        if z[j + 1] == 0:  # Stop if ion leaves the simulation area
            break
        ax.plot(z[j:j + 2], r[j:j + 2], color=colors[j], linewidth=0.5)

plt.colorbar(im, ax=ax, label='|E| (V/cm)')
ax.set_xlabel("z (cm)")
ax.set_ylabel("r (cm)")
ax.set_title("Ar⁺-Ion Trajectory Simulation")
ax.set_xlim(0, z_max)
ax.set_ylim(-r_max / 2, r_max / 2)
plt.tight_layout()
print("Simulation finished.")
plt.show()
