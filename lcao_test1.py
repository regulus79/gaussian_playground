from new_guassian_tools import *
from contracted_gaussian_tools import *
from lcao_tools import *
from visualize_tools import *
import numpy as np

num_trials = 20

spacings = np.linspace(0.5 * bohr_radius, 4 * bohr_radius, num_trials)

orbitalEnergyBonding = np.zeros(num_trials)
orbitalEnergyAntiBonding = np.zeros(num_trials)
nucleiEnergy = np.zeros(num_trials)

exponent = 0.1 / bohr_radius**2

for i, spacing in enumerate(spacings):
	pos1 = np.array([0,0,0])
	contractedGaussian1a = ContractedGaussian([CartesianGaussian(pos1, exponent, np.array([0,0,0]))], [1])
	contractedGaussian1b = ContractedGaussian([CartesianGaussian(pos1, 2*exponent, np.array([0,0,0]))], [1])
	contractedGaussian1c = ContractedGaussian([CartesianGaussian(pos1, 4*exponent, np.array([0,0,0]))], [1])
	contractedGaussian1d = ContractedGaussian([CartesianGaussian(pos1, 8*exponent, np.array([0,0,0]))], [1])
	contractedGaussian1e = ContractedGaussian([CartesianGaussian(pos1, 16*exponent, np.array([0,0,0]))], [1])
	pos2 = np.array([spacing,0,0])
	contractedGaussian2a = ContractedGaussian([CartesianGaussian(pos2, exponent, np.array([0,0,0]))], [1])
	contractedGaussian2b = ContractedGaussian([CartesianGaussian(pos2, 2*exponent, np.array([0,0,0]))], [1])
	contractedGaussian2c = ContractedGaussian([CartesianGaussian(pos2, 4*exponent, np.array([0,0,0]))], [1])
	contractedGaussian2d = ContractedGaussian([CartesianGaussian(pos2, 8*exponent, np.array([0,0,0]))], [1])
	contractedGaussian2e = ContractedGaussian([CartesianGaussian(pos2, 16*exponent, np.array([0,0,0]))], [1])

	orbitals = [contractedGaussian1a, contractedGaussian1b, contractedGaussian1c, contractedGaussian1d, contractedGaussian1e, contractedGaussian2a, contractedGaussian2b, contractedGaussian2c, contractedGaussian2d, contractedGaussian2e]
	nuclei = [Nucleus(pos1, 1), Nucleus(pos2, 1)]

	print("----", i, spacing)
	eigenvalues, eigenvectors = orbtialEigs(nuclei, orbitals, 0.01 * bohr_radius)
	print(np.sort(eigenvalues) / charge_e)
	minIndex = np.argmin(eigenvalues)
	maxIndex = np.argsort(eigenvalues)[1]
	#print(spacing, eigenvalues[minIndex], eigenvectors[minIndex])
	orbitalEnergyBonding[i] = eigenvalues[minIndex] #* 2
	orbitalEnergyAntiBonding[i] = eigenvalues[maxIndex] #* 2
	nucleiEnergy[i] = nucleiRepulsionEnergy(nuclei)

totalEnergy = orbitalEnergyBonding + nucleiEnergy

print(f"Min energy {np.min(totalEnergy)} at distance {spacings[np.argmin(totalEnergy)]}")
print(orbitalEnergyBonding)
print(orbitalEnergyAntiBonding)
print(nucleiEnergy)

plt.plot(spacings, orbitalEnergyBonding / charge_e, label = "Bonding Orbital energy (eV)")
plt.plot(spacings, orbitalEnergyAntiBonding / charge_e, label = "Antibonding Orbital energy (eV)")
plt.plot(spacings, nucleiEnergy / charge_e, label = "Nuclei energy (eV)")
plt.plot(spacings, totalEnergy / charge_e, label = "Total energy (bonding) (eV)")
plt.plot(spacings, (orbitalEnergyAntiBonding + nucleiEnergy) / charge_e, label = "Total energy (antibonding) (eV)")
plt.legend()
plt.ylabel("Total Energy")
plt.xlabel("Bond Length")
plt.show()