from new_guassian_tools import *
from contracted_gaussian_tools import *
from lcao_tools import *
from visualize_tools import *
import numpy as np

num_trials = 20

spacings = np.linspace(0.001 * bohr_radius, 0.1 * bohr_radius, num_trials)

orbitalEnergy = np.zeros(num_trials)
nucleiEnergy = np.zeros(num_trials)

exponent = 1e7 / bohr_radius

for i, spacing in enumerate(spacings):
	pos1 = np.array([0,0,0])
	cartesianGaussian1 = CartesianGaussian(pos1, exponent, np.array([0,0,0]))
	contractedGaussian1 = ContractedGaussian([cartesianGaussian1], [1])
	pos2 = np.array([spacing,0,0])
	cartesianGaussian2 = CartesianGaussian(pos2, exponent, np.array([0,0,0]))
	contractedGaussian2 = ContractedGaussian([cartesianGaussian2], [1])

	orbitals = [contractedGaussian1, contractedGaussian2]
	nuclei = [Nucleus(pos1, 1), Nucleus(pos2, 1)]

	eigenvalues, eigenvectors = orbtialEigs(nuclei, orbitals, 0.01)
	#print(eigenvalues, eigenvectors)
	minIndex = np.argmin(eigenvalues)
	#print(spacing, eigenvalues[minIndex], eigenvectors[minIndex])
	orbitalEnergy[i] = eigenvalues[minIndex]
	nucleiEnergy[i] = nucleiRepulsionEnergy(nuclei)

totalEnergy = orbitalEnergy + nucleiEnergy

print(f"Min energy {np.min(totalEnergy)} at distance {spacings[np.argmin(totalEnergy)]}")

plt.plot(spacings, orbitalEnergy, label = "Orbital energy")
plt.plot(spacings, nucleiEnergy, label = "Nuclei energy")
plt.plot(spacings, totalEnergy, label = "Total energy")
plt.legend()
plt.ylabel("Total Energy")
plt.xlabel("Bond Length")
plt.show()