
from new_guassian_tools import *
from contracted_gaussian_tools import *

class Nucleus():
	def __init__(self, pos, charge):
		self.pos = pos
		self.charge = charge

def primativeHamiltonianIntegral(orbital1, orbital2, nuclei, steplength):
	kinetic = primativeKineticIntegral(orbital1, orbital2, steplength)
	potential = 0
	for nucleus in nuclei:
		potential += -nucleus.charge * primativeCoulombIntegral(orbital1, orbital2, nucleus.pos)
	#print("kinetic", kinetic, "potential", potential)
	return kinetic + potential


def coulombMatrix(nuclei, orbitals, steplength):
	H = np.zeros((len(orbitals), len(orbitals)))
	for i, orbital1 in enumerate(orbitals):
		for j, orbital2 in enumerate(orbitals):
			H[i][j] = evalTwoOrbitalFunctionWithContractedGaussians(primativeHamiltonianIntegral, orbital1, orbital2, steplength, nuclei, steplength)
	return H


def overlapMatrix(orbitals, steplength):
	S = np.zeros((len(orbitals), len(orbitals)))
	for i, orbital1 in enumerate(orbitals):
		for j, orbital2 in enumerate(orbitals):
			S[i][j] = evalTwoOrbitalFunctionWithContractedGaussians(primativeOverlapIntegral, orbital1, orbital2, steplength)
	return S

def orbtialEigs(nuclei, orbitals, steplength):
	S = overlapMatrix(orbitals, steplength)
	#print(S)
	H = coulombMatrix(nuclei, orbitals, steplength)
	#print(H)
	A = np.linalg.inv(S) @ H
	eigenvalues, eigenvectors = np.linalg.eig(A)
	return eigenvalues, np.transpose(eigenvectors)

def nucleiRepulsionEnergy(nuclei):
	total = 0
	for i in range(len(nuclei)):
		for j in range(len(nuclei)):
			if i < j:
				total += nuclei[i].charge * nuclei[j].charge / np.sum((nuclei[i].pos - nuclei[j].pos)**2)**0.5  *  charge_e**2 / (4 * math.pi * epsilon0)
	return total

# Assuming # protons = # electrons
def occupiedElectronEnergy(eigenvalues, nuclei):
	total_electrons = sum([n.charge for n in nuclei])
	sorted_indicies = np.argsort(eigenvalues)
	occupied_indicies = sorted_indicies[0:total_electrons//2]
	total = 0
	if total_electrons % 2 == 0:
		total += 2 * np.sum(eigenvalues[occupied_indicies])
	else:
		total += 2 * np.sum(eigenvalues[occupied_indicies]) - eigenvalues[occupied_indicies[-1]]
	return total
