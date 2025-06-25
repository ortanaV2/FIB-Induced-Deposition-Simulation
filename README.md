# Focused-Ion-Beam (FIB) Simulation
This project simulates the motion of Ar⁺ ions through a system of electrostatic lenses and apertures. The electric field is calculated by solving the Laplace equation on a 2D grid (r-z), and ion trajectories are computed based on the resulting field.

## Features
- Electrostatic potential calculation using a finite difference Laplace solver
- Electric field computation from the potential
- Ion trajectory simulation with random emission angles
- Aperture collision handling

![Focused-Ion-Beam-Figure](https://github.com/user-attachments/assets/835186fc-0633-4d57-88d8-87878a114557)

## Requirements
- Python ≥ 3.8
- NumPy
- Matplotlib
- Numba

### Installation
```bash
pip install numpy matplotlib numba
```

## Grid Configuration
The simulation space is defined in cylindrical coordinates:
```python
Nx, Nz = 300, 800
dr, dz = 0.01, 0.01  # grid resolution in cm
r_max, z_max = Nx * dr, Nz * dz
```

## Electrodes and Apertures (Lens Configuration)
### Lenses
Defined by position ``z``, axial length ``depth``, inner radius ``r_in``, outer radius ``r_out``, and potential ``V``.
```python
lenses = {
    "Accelerator_Lens_1": {"z": 1.7, "depth": 0.3, "r_in": 0.2, "r_out": 1.0, "V": -30000},
    ...
}
```
### Apertures
Defined as non-conductive barriers (e.g. PVC) with similar geometry, but no voltage. These block ions when collided with:
```python
apertures = {
    "Aperture1": {"z": 0.7, "depth": 0.1, "r_in": 0.05, "r_out": 1.0}
}
```

## In Depth: Simulation Steps
### Laplace Solver
The Laplace equation is solved iteratively over the grid with Dirichlet boundary conditions given by the lens voltages:
```python
phi_new[1:-1, 1:-1] = 0.25 * (phi[2:, 1:-1] + phi[:-2, 1:-1] +
                              phi[1:-1, 2:] + phi[1:-1, :-2])
```
Lens regions are held at constant potential throughout the iteration.

### Electric Field Calculation
The electric field is calculated from the gradient of the potential:
```python
Er = -(phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dr)
Ez = -(phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dz)
```

### Ion Trajectory Simulation
Each ion is initialized with a random angle and radial offset. The motion is integrated using the electric field:
```python
vr += ar * dt
vz += az * dt
r += vr * dt
z += vz * dt
```
Ion paths terminate when:
- They leave the simulation domain
- They hit a defined aperture
