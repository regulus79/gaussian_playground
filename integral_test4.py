
from new_guassian_tools import *
from contracted_gaussian_tools import *
from lcao_tools import *
from visualize_tools import *
import numpy as np
import math

pos1 = np.array([0,0,0])
exponent1 = 1
pos2 = np.array([0,0,0])
exponent2 = 1
pos3 = np.array([1,0,0])
exponent3 = 1
pos4 = np.array([1,0,0])
exponent4 = 1

# Fourier transform of gaussian
# 1/(2pi) * sqrt(pi/b) * exp(-k^2/(4b)) * exp(-ikc)
# in 3d:
# 1/(2pi)^3 * sqrt(pi/b) * exp(-k^2/(4b)) * exp(-ikc)

# Fourier transform of 1/r
# 4pi/k^2



# Fourier method

num_points = 50
points_range = 10
dx = 2*points_range / num_points
coords_1d = np.linspace(-points_range, points_range, num_points).astype("complex")
kx, ky, kz = np.meshgrid(coords_1d, coords_1d, coords_1d)
print(f"Fourier, num axis points: {num_points}, axis range: {points_range}")

# Between 1 and 2:
fourier1 = 1/(2*np.pi)**3 * np.sqrt(np.pi / exponent1) * np.exp(-(kx**2 + ky**2 + kz**2) / (4 * exponent1)) * np.exp(-1j * (kx*pos1[0] + ky*pos1[1] + kz*pos1[2]))
fourier2 = 1/(2*np.pi)**3 * np.sqrt(np.pi / exponent2) * np.exp(-(kx**2 + ky**2 + kz**2) / (4 * exponent2)) * np.exp(-1j * (kx*pos2[0] + ky*pos2[1] + kz*pos2[2]))
fourierPotential = 4 * np.pi / (kx**2 + ky**2 + kz**2)
total = np.sum(fourier1 * fourier2 * fourierPotential * dx**3)

print(total)


# Cartesian method
# int over x1, int over x2
# f(x1)w(x2-x1)g(x2)

num_points = 10
points_range = 5
dx = 2*points_range / num_points
coords_1d = np.linspace(-points_range, points_range, num_points)
x1, y1, z1 = np.meshgrid(coords_1d, coords_1d, coords_1d)
print(f"Cartesian, num axis points: {num_points}, axis range: {points_range}")
nudge = 0.5 * dx

total = 0
func1 = np.exp(-exponent1*((pos1[0] - x1)**2 + (pos1[1] - y1)**2 + (pos1[2] - z1)**2))
i = 0
for x2 in coords_1d:
	for y2 in coords_1d:
		for z2 in coords_1d:
			i+=1
			print(i, i / (num_points**3), end="\r")
			weight = 1 / ((x1 - x2 + nudge)**2 + (y1 - y2 + nudge)**2 + (z1 - z2 + nudge)**2)**0.5
			func2 = np.exp(-exponent2*((pos2[0] - x2)**2 + (pos2[1] - y2)**2 + (pos2[2] - z2)**2))
			total += np.sum(func1 * weight * func2) * dx**3 * dx**3
print("\nDone", total)

total = 0
i = 0
for x in coords_1d:
	for y in coords_1d:
		for z in coords_1d:
			for x2 in coords_1d:
				for y2 in coords_1d:
					for z2 in coords_1d:
						i+=1
						print(i, i / (num_points**6), end="\r")
						func1 = math.exp(-exponent1*((pos1[0] - x)**2 + (pos1[1] - y)**2 + (pos1[2] - z)**2))
						weight = 1 / ((x - x2 + nudge)**2 + (y - y2 + nudge)**2 + (z - z2 + nudge)**2)**0.5
						func2 = math.exp(-exponent2*((pos2[0] - x2)**2 + (pos2[1] - y2)**2 + (pos2[2] - z2)**2))
						total += func1 * weight * func2 * dx**3 * dx**3

print("\nDone2", total)