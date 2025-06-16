import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Define grid size and resolution
Nx, Nz = 300, 600
dr, dz = 0.01, 0.01  # in cm
r_max, z_max = Nx * dr, Nz * dz

# Focus-Linsen Definition
    # "Focus": {
    #     "z": 0.95, "depth": 0.5, "V": 0,
    #     "r_in_start": 0.3, "r_in_end": 0.2,
    #     "r_out_start": 0.5, "r_out_end": 0.4
    # },

lenses = {
    "Extractor": {"z": 0.3, "depth": 0.5, "r_in": 0.2, "r_out": 0.4, "V": -30000},
    "Lens1": {"z": 0.95, "depth": 0.5, "r_in": 0.3, "r_out": 0.5, "V": 0},
    "Lens2": {"z": 1.6, "depth": 0.5, "r_in": 0.2, "r_out": 0.4, "V": -30000},
    "Lens3": {"z": 2.35, "depth": 0.5, "r_in": 0.2, "r_out": 0.4, "V": 0},
}

# Potential initialisieren
phi = np.zeros((Nx, Nz))
center = Nx // 2

for name, linse in lenses.items():
    z0 = int(linse["z"] / dz)
    z1 = int((linse["z"] + linse["depth"]) / dz)
    if name == "Focus":
        for zi in range(z0, z1):
            frac = (zi - z0) / (z1 - z0)
            r_in = int(((1 - frac) * linse["r_in_start"] + frac * linse["r_in_end"]) / dr)
            r_out = int(((1 - frac) * linse["r_out_start"] + frac * linse["r_out_end"]) / dr)
            phi[center + r_in:center + r_out, zi] = linse["V"]
            phi[center - r_out:center - r_in, zi] = linse["V"]
    else:
        ri = int(linse["r_in"] / dr)
        ro = int(linse["r_out"] / dr)
        phi[center + ri:center + ro, z0:z1] = linse["V"]
        phi[center - ro:center - ri, z0:z1] = linse["V"]

# Laplace Solver
phi_new = phi.copy()
for iteration in range(3000):
    phi_new[1:-1, 1:-1] = 0.25 * (phi[2:, 1:-1] + phi[:-2, 1:-1] +
                                  phi[1:-1, 2:] + phi[1:-1, :-2])
    for name, linse in lenses.items():
        z0 = int(linse["z"] / dz)
        z1 = int((linse["z"] + linse["depth"]) / dz)
        if name == "Focus":
            for zi in range(z0, z1):
                frac = (zi - z0) / (z1 - z0)
                r_in = int(((1 - frac) * linse["r_in_start"] + frac * linse["r_in_end"]) / dr)
                r_out = int(((1 - frac) * linse["r_out_start"] + frac * linse["r_out_end"]) / dr)
                phi_new[center + r_in:center + r_out, zi] = linse["V"]
                phi_new[center - r_out:center - r_in, zi] = linse["V"]
        else:
            ri = int(linse["r_in"] / dr)
            ro = int(linse["r_out"] / dr)
            phi_new[center + ri:center + ro, z0:z1] = linse["V"]
            phi_new[center - ro:center - ri, z0:z1] = linse["V"]
    phi, phi_new = phi_new, phi
    if iteration % 500 == 0:
        print(f"Laplace iteration {iteration}/3000")

print("Laplace solver completed.")

# Elektrisches Feld berechnen
Er = -(phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dr)
Ez = -(phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dz)
r_vals = (np.arange(1, Nx - 1) - Nx // 2) * dr
z_vals = np.arange(1, Nz - 1) * dz

@njit
def bilinear_interpolate(r_vals, z_vals, field, r, z):
    dr = r_vals[1] - r_vals[0]
    dz = z_vals[1] - z_vals[0]
    i = int((r - r_vals[0]) / dr)
    j = int((z - z_vals[0]) / dz)
    if i < 0 or i >= field.shape[0] - 1 or j < 0 or j >= field.shape[1] - 1:
        return 0.0
    r1, z1 = r_vals[i], z_vals[j]
    fr1z1 = field[i, j]
    fr2z1 = field[i + 1, j]
    fr1z2 = field[i, j + 1]
    fr2z2 = field[i + 1, j + 1]
    t = (r - r1) / dr
    u = (z - z1) / dz
    return (1 - t) * (1 - u) * fr1z1 + t * (1 - u) * fr2z1 + (1 - t) * u * fr1z2 + t * u * fr2z2

@njit(parallel=True)
def simulate_ions(n_ionen, steps, dt, r_vals, z_vals, Er, Ez, q, m, v0, r0_max):
    results_r = np.zeros((n_ionen, steps))
    results_z = np.zeros((n_ionen, steps))
    results_vr = np.zeros((n_ionen, steps))
    results_vz = np.zeros((n_ionen, steps))
    for i in prange(n_ionen):
        angle = np.deg2rad(np.random.uniform(-90, 90))
        r = np.random.uniform(-r0_max, r0_max)
        z = 0.0
        vr = v0 * np.sin(angle)
        vz = v0 * np.cos(angle)
        for t in range(steps):
            results_r[i, t] = r
            results_z[i, t] = z
            results_vr[i, t] = vr
            results_vz[i, t] = vz
            Er_loc = bilinear_interpolate(r_vals, z_vals, Er, r, z)
            Ez_loc = bilinear_interpolate(r_vals, z_vals, Ez, r, z)
            ar = (q / m) * Er_loc * 1e2
            az = (q / m) * Ez_loc * 1e2
            vr += ar * dt
            vz += az * dt
            r += vr * dt
            z += vz * dt
            if z >= z_vals[-1] or r < r_vals[0] or r > r_vals[-1]:
                break
        if i % 5 == 0:
            print(f"Ion {i + 1}/{n_ionen} simulated")
    return results_r, results_z, results_vr, results_vz

# Simulation starten
q, m = 1.602e-19, 40 * 1.67e-27
n_ionen = 40
v0 = 1e5  # cm/s
r0_max = 0.05
steps = 2000
dt = 1e-9

print("Starting ion simulation...")
results_r, results_z, results_vr, results_vz = simulate_ions(
    n_ionen, steps, dt, r_vals, z_vals, Er, Ez, q, m, v0, r0_max
)
print("Ion simulation completed.")

# Plotten
fig, ax = plt.subplots(figsize=(10, 6))
extent = [0, z_max, -r_max / 2, r_max / 2]
E_mag = np.sqrt(Er**2 + Ez**2)
im = ax.imshow(E_mag, cmap='jet', origin='lower', extent=extent, aspect='auto')

for i in range(n_ionen):
    r = results_r[i]
    z = results_z[i]
    vr = results_vr[i]
    vz = results_vz[i]
    speed = np.sqrt(vr**2 + vz**2)
    norm_speed = (speed - speed.min()) / (speed.max() - speed.min() + 1e-9)
    colors = plt.cm.twilight_shifted(norm_speed)
    for j in range(steps - 1):
        if z[j + 1] == 0:
            break
        ax.plot(z[j:j + 2], r[j:j + 2], color=colors[j], linewidth=0.5)

plt.colorbar(im, ax=ax, label='|E| (V/cm)')
ax.set_xlabel("z (cm)")
ax.set_ylabel("r (cm)")
ax.set_title("Ar⁺-Ion Trajectory Simulation (mit geneigter Linse 1)")
ax.set_xlim(0, z_max)
ax.set_ylim(-r_max / 2, r_max / 2)
plt.tight_layout()
plt.savefig("result.png", dpi=300)
print("Plot saved.")
