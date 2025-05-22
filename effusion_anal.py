
#!/usr/bin/env python3
# Written by Edward Troccoli, 2025

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cfpack as cfp

# Argument parser setup
parser = argparse.ArgumentParser(description="Plot different quantities from simulation data.")
parser.add_argument("-v", "--variable", required=True, help="Variable to plot")
# Parse the command-line arguments
args = parser.parse_args()


file_path = "../PHYS2020SimProj/simulation_results.xlsx"
cfp.print('Reading:'+file_path,color='green')
xls = pd.ExcelFile(file_path)

# Read the sheets
df_main = pd.read_excel(xls, sheet_name='Simulation Data')
df_escape = pd.read_excel(xls, sheet_name='Escape Times')

if args.variable == 'energy':
    # Plotting Energy vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(df_main['Time'], df_main['Kinetic Energy'], label='Kinetic Energy')
    plt.plot(df_main['Time'], df_main['Potential Energy'], label='Potential Energy')
    plt.plot(df_main['Time'], df_main['Total Energy'], label='Total Energy', linewidth=2, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy vs Time")
    plt.legend()
    plt.show()

if args.variable == 'escapes':
    # Plotting Number of Escaped Particles vs Time
    cfp.plot(x=df_main['Time'], y=30-df_main['Escaped Particles'], xlabel="Time", ylabel="Escaped Particles", label='Cumulative Escaped Particles', color='green')
    plt.show()
if args.variable == 'histo':
    # Optional: Histogram of Escape Times
    plt.figure(figsize=(10, 6))
    plt.hist(df_escape['Escape Time'], bins=30, color='orange', edgecolor='black')
    plt.xlabel("Escape Time")
    plt.ylabel("Number of Particles")
    plt.title("Histogram of Escape Times")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
