import json
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Define grid size and resolution
Nx, Nz = 300, 800
dr, dz = 0.01, 0.01  # in cm
r_max, z_max = Nx * dr, Nz * dz

# Lenses and apertures
lenses = {
    "Extractor": {"z": 0.3, "depth": 0.3, "r_in": 0.2, "r_out": 1.0, "V": -30000},
    "Electrode": {"z": 1.0, "depth": 0.3, "r_in": 0.3, "r_out": 1.0, "V": 0},
    "Accelerator_Lens_1": {"z": 1.7, "depth": 0.3, "r_in": 0.2, "r_out": 1.0, "V": -30000},
    "Accelerator_Lens_2": {"z": 2.2, "depth": 0.3, "r_in": 0.3, "r_out": 1.0, "V": 0},
    "Accelerator_Lens_3": {"z": 2.7, "depth": 0.3, "r_in": 0.15, "r_out": 1.0, "V": -30000},
    "Condensor_Lens_1": {"z": 3.2, "depth": 0.3, "r_in": 0.25, "r_out": 1.0, "V": 0},
    "Stock": {"z": 3.7, "depth": 0.2, "r_in": 0, "r_out": 1, "V": -30000},
    # "Condensor_Lens_2": {
    #     "z": 3.9, "depth": 0.3, "V": -30000,
    #     "r_in_start": 0.1, "r_in_end": 0.05,
    #     "r_out_start": 1.0, "r_out_end": 1.0
    # },
}

apertures = {
    "Aperture1": {"z": 0.7, "depth": 0.1, "r_in": 0.05, "r_out": 1.0}
}

print(f"Configuration:\n{json.dumps(lenses, indent=4)}")

# Initialize potential
phi = np.zeros((Nx, Nz))
center = Nx // 2

for name, lense in lenses.items():
    z0 = int(lense["z"] / dz)
    z1 = int((lense["z"] + lense["depth"]) / dz)
    if "r_in_start" in lense:
        for zi in range(z0, z1):
            frac = (zi - z0) / (z1 - z0)
            r_in = int(((1 - frac) * lense["r_in_start"] + frac * lense["r_in_end"]) / dr)
            r_out = int(((1 - frac) * lense["r_out_start"] + frac * lense["r_out_end"]) / dr)
            phi[center + r_in:center + r_out, zi] = lense["V"]
            phi[center - r_out:center - r_in, zi] = lense["V"]
    else:
        ri = int(lense["r_in"] / dr)
        ro = int(lense["r_out"] / dr)
        phi[center + ri:center + ro, z0:z1] = lense["V"]
        phi[center - ro:center - ri, z0:z1] = lense["V"]

# Laplace solver
phi_new = phi.copy()
for iteration in range(3000):
    phi_new[1:-1, 1:-1] = 0.25 * (phi[2:, 1:-1] + phi[:-2, 1:-1] +
                                  phi[1:-1, 2:] + phi[1:-1, :-2])
    for name, lense in lenses.items():
        z0 = int(lense["z"] / dz)
        z1 = int((lense["z"] + lense["depth"]) / dz)
        if "r_in_start" in lense:
            for zi in range(z0, z1):
                frac = (zi - z0) / (z1 - z0)
                r_in = int(((1 - frac) * lense["r_in_start"] + frac * lense["r_in_end"]) / dr)
                r_out = int(((1 - frac) * lense["r_out_start"] + frac * lense["r_out_end"]) / dr)
                phi_new[center + r_in:center + r_out, zi] = lense["V"]
                phi_new[center - r_out:center - r_in, zi] = lense["V"]
        else:
            ri = int(lense["r_in"] / dr)
            ro = int(lense["r_out"] / dr)
            phi_new[center + ri:center + ro, z0:z1] = lense["V"]
            phi_new[center - ro:center - ri, z0:z1] = lense["V"]
    phi, phi_new = phi_new, phi
    if iteration % 500 == 0:
        print(f"Laplace iteration {iteration}/3000")

print("Laplace solver completed.")

# Calculate electric field
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

aperture_array = np.zeros((Nx, Nz), dtype=np.uint8)
for aperture in apertures.values():
    z0 = int(aperture["z"] / dz)
    z1 = int((aperture["z"] + aperture["depth"]) / dz)
    ri = int(aperture["r_in"] / dr)
    ro = int(aperture["r_out"] / dr)
    aperture_array[center + ri:center + ro, z0:z1] = 1
    aperture_array[center - ro:center - ri, z0:z1] = 1

@njit(parallel=True)
def simulate_ions(n_ionen, steps, dt, r_vals, z_vals, Er, Ez, q, m, v0, r0_max, aperture_array):
    Nx, Nz = aperture_array.shape
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
            ri = int((r - r_vals[0]) / dr)
            zi = int((z - z_vals[0]) / dz)
            if z >= z_vals[-1] or r < r_vals[0] or r > r_vals[-1]:
                break
            if ri >= 0 and ri < Nx and zi >= 0 and zi < Nz:
                if aperture_array[ri, zi] == 1:
                    break
    return results_r, results_z, results_vr, results_vz

# Start simulation
q, m = 1.602e-19, 40 * 1.67e-27
n_ionen = 200
v0 = 1e5  # cm/s
r0_max = 0.05
steps = 2500
dt = 1e-9

print("Starting ion simulation...")
results_r, results_z, results_vr, results_vz = simulate_ions(
    n_ionen, steps, dt, r_vals, z_vals, Er, Ez, q, m, v0, r0_max, aperture_array
)
print("Ion simulation completed.")
print("Plotting...")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
extent = [0, z_max, -r_max / 2, r_max / 2]
E_mag = np.sqrt(Er**2 + Ez**2)

norm_E = (E_mag - E_mag.min()) / (np.ptp(E_mag) + 1e-9)
colored_E = plt.cm.jet(norm_E)  # RGBA

aperture_mask = np.zeros_like(E_mag)
for aperture in apertures.values():
    z0 = int(aperture["z"] / dz) - 1
    z1 = int((aperture["z"] + aperture["depth"]) / dz) + 1
    ri = int(aperture["r_in"] / dr)
    ro = int(aperture["r_out"] / dr)
    aperture_mask[center + ri:center + ro, z0:z1] = 1
    aperture_mask[center - ro:center - ri, z0:z1] = 1

r_col, g_col, b_col = 255/255, 244/255, 214/255  # ≈ (1.0, 0.957, 0.839)
for i in range(3):  # R, G, B
    colored_E[:, :, i][aperture_mask == 1] = [r_col, g_col, b_col][i]
colored_E[:, :, 3][aperture_mask == 1] = 1.0

im = ax.imshow(colored_E, origin='lower', extent=extent, aspect='auto')

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
        ax.plot(z[j:j + 2], r[j:j + 2], color=colors[j], linewidth=0.1)

plt.colorbar(im, ax=ax, label='|E| (V/cm)')
ax.set_xlabel("z (cm)")
ax.set_ylabel("r (cm)")
ax.set_title("Ar⁺-Ion Trajectory Simulation")
ax.set_xlim(0, z_max)
ax.set_ylim(-r_max / 2, r_max / 2)
plt.tight_layout()
# plt.savefig("result.png", dpi=300)
plt.show()
print("Plot saved.")
