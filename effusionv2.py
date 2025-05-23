#!/usr/bin/env python3
# Written by Edward Troccoli, 2025
# Inspiration from Louis Sharma, 2023

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import cfpack as cfp
import os

#---------------------Initializing Parameters-----------------------------------

N = 50  # number of particles
W, H = 10, 10  # box dimensions
y_pos = H / 2     
hole_height = 0.5         
x_pos = W                

T = 500000.0
m = 1.0
k = 1.0
v0 = np.sqrt(k * T / m)

radius = 0.15
current_time = 0.0
dt = 1e-5
steps = 50000

escaped = np.zeros(N, dtype=bool)
escape_times = []
escape_count = 0

# Energy tracking lists
time_list, KE_list, PE_list, total_energy_list, escaped_count_list = [], [], [], [], []

out_path = '../PHYS2020SimProj/SimResults/'

if not os.path.isdir(out_path):
        cfp.run_shell_command('mkdir '+out_path)

#---------------------Lennard-Jones Gas Class-----------------------------------

class LJ_Gas:

    def __init__(self, N, m, radius, W, H, v0, dt, steps):
        self.N = N 
        self.m = m 
        self.radius = radius 
        self.H = H 
        self.W = W 
        self.dt = dt 
        self.steps = steps 
        self.time = dt * steps 
        self.v0 = v0 

        self.r = np.stack((
            np.random.uniform(radius, W - radius, N),
            np.random.uniform(radius, H - radius, N)
        ), axis=1)

        vx = np.random.normal(0, v0, N)
        vy = np.random.normal(0, v0, N)
        self.v = np.stack((vx, vy), axis=1)

    def compute_energies(self):
        # Kinetic Energy
        KE = 0.5 * self.m * np.sum(self.v**2)

        # Lennard-Jones Potential Energy
        PE = 0.0
        sigma = 1.0
        epsilon = 1.0
        for i in range(len(self.r)):
            for j in range(i + 1, len(self.r)):
                r_ij = np.linalg.norm(self.r[i] - self.r[j])
                if r_ij > 1e-5:  # avoid singularity
                    PE += 4 * epsilon * ((sigma / r_ij)**12 - (sigma / r_ij)**6)
        return KE, PE

    def lennard_jones_potential(self):
        return np.zeros_like(self.r)  # Replace with actual force if needed

    def check_collisions(self):
        force = self.lennard_jones_potential()
        r_next = self.r + self.v * self.dt + (force / (2 * self.m)) * self.dt**2

        # Walls
        mask_left = self.r[:, 0] < self.radius
        self.v[mask_left, 0] *= -1
        self.r[mask_left, 0] = self.radius

        mask_bottom = self.r[:, 1] < self.radius
        self.v[mask_bottom, 1] *= -1
        self.r[mask_bottom, 1] = self.radius

        mask_top = self.r[:, 1] > self.H - self.radius
        self.v[mask_top, 1] *= -1
        self.r[mask_top, 1] = self.H - self.radius

        mask_right = (self.r[:, 0] > self.W - self.radius) & (
            (self.r[:, 1] < y_pos - hole_height / 2) | (self.r[:, 1] > y_pos + hole_height / 2)
        )
        self.v[mask_right, 0] *= -1
        self.r[mask_right, 0] = self.W - self.radius

        n_particles = len(self.r)
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                if np.linalg.norm(r_next[i] - r_next[j]) < 2 * self.radius:
                    rdiff = self.r[i] - self.r[j]
                    vdiff = self.v[i] - self.v[j]
                    self.v[i] -= np.dot(rdiff, vdiff) / np.dot(rdiff, rdiff) * rdiff
                    self.v[j] += np.dot(rdiff, vdiff) / np.dot(rdiff, rdiff) * rdiff

    def check_escaped(self, escaped_mask, escape_times, escape_count, current_time):
        escape_condition = (
            (self.r[:, 0] > self.W) &
            (self.r[:, 1] > y_pos - hole_height / 2) &
            (self.r[:, 1] < y_pos + hole_height / 2) &
            (~escaped_mask)
        )

        newly_escaped = np.sum(escape_condition)
        escape_count += newly_escaped
        escape_times.extend([current_time] * newly_escaped)

        keep_mask = ~escape_condition
        self.r = self.r[keep_mask]
        self.v = self.v[keep_mask]
        escaped_mask = escaped_mask[keep_mask]

        return escaped_mask, escape_times, escape_count

    def step(self):
        self.check_collisions()
        force = self.lennard_jones_potential()
        self.r += self.v * self.dt + (force / (2 * self.m)) * self.dt**2
        updated_force = self.lennard_jones_potential()
        self.v += ((updated_force + force) / (2 * self.m)) * self.dt

#---------------------Creating the animation------------------------------------------

gas = LJ_Gas(N, m, radius, W, H, v0, dt, steps)

fig, ax = plt.subplots()
scat = ax.scatter(gas.r[:, 0], gas.r[:, 1], s=5)
time_text = ax.text(0.02, 1.02, 'Time: '+str(current_time), transform=ax.transAxes)
escape_text = ax.text(0.02, 1.10, 'Number of escaped particles: '+str(escape_count), transform=ax.transAxes)
ax.set_xlim(0, gas.W)
ax.set_ylim(0, gas.H)
ax.plot([W, W], [y_pos - hole_height/2, y_pos + hole_height/2], color='red', linewidth=10)

def update(frame):
    global current_time, escape_count, escaped, escape_times
    global time_list, KE_list, PE_list, total_energy_list, escaped_count_list

    if frame % 50 == 0:
        cfp.print(f"Frame {frame} of {steps}",color='magenta')  # <- Added line

    gas.step()
    current_time += dt
    escaped, escape_times, escape_count = gas.check_escaped(escaped, escape_times, escape_count, current_time)

    # Energy calculations
    KE, PE = gas.compute_energies()
    total_E = KE + PE
    time_list.append(current_time)
    KE_list.append(KE)
    PE_list.append(PE)
    total_energy_list.append(total_E)
    escaped_count_list.append(escape_count)

    # Update plot
    scat.set_offsets(gas.r)
    time_text.set_text(f"Time: {current_time:.2f}")
    escape_text.set_text(f"Number of escaped particles: {escape_count}")
    return scat, time_text, escape_text

ani = animation.FuncAnimation(fig, update, frames=steps, interval=0.1, blit=False)
ani.save(out_path+f"escaped_particles_{N}_{steps}.mp4", writer="ffmpeg", fps=30)

constants = {
    'Parameter': ['N (particles)', 'T (K)', 'm (mass)', 'k (Boltzmann constant)', 
                  'Box Width (W)', 'Box Height (H)', 'Hole Height', 'Time Step (dt)', 
                  'Steps', 'Initial Speed v0', 'Particle Radius'],
    'Value': [N, T, m, k, W, H, hole_height, dt, steps, v0, radius]
}
df_constants = pd.DataFrame(constants)

# Main simulation data
df_main = pd.DataFrame({
    'Time': time_list,
    'Kinetic Energy': KE_list,
    'Potential Energy': PE_list,
    'Total Energy': total_energy_list,
    'Escaped Particles': escaped_count_list
})

# Escape times as a separate DataFrame
df_escape = pd.DataFrame({'Escape Time': escape_times})

# Save both to an Excel file with two sheets
with pd.ExcelWriter(out_path+"simulation_results_{N}_{steps}.xlsx", engine='xlsxwriter') as writer:
    df_main.to_excel(writer, sheet_name='Simulation Data', index=False)
    df_escape.to_excel(writer, sheet_name='Escape Times', index=False)
    df_constants.to_excel(writer, sheet_name='Simulation Parameters', index=False)