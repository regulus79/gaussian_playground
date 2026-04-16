
from new_guassian_tools import *
from contracted_gaussian_tools import *

class Nucleus():
	def __init__(self, pos, charge):
		self.pos = pos
		self.charge = charge

def primativeHamiltonianIntegral(orbital1, orbital2, potentialPos, potentialMagnitude, steplength):
	return -1/2 * primativeKineticIntegral(orbital1, orbital2, steplength) + potentialMagnitude * primativeCoulombIntegral(orbital1, orbital2, potentialPos)


def coulombMatrix(nuclei, orbitals, steplength):
	H = np.zeros((len(orbitals), len(orbitals)))
	for i, orbital1 in enumerate(orbitals):
		for j, orbital2 in enumerate(orbitals):
			total = 0
			for nucleus in nuclei:
				total += evalTwoOrbitalFunctionWithContractedGaussians(primativeHamiltonianIntegral, orbital1, orbital2, steplength, nucleus.pos, nucleus.charge, steplength)
			H[i][j] = total
	return H


def overlapMatrix(orbitals, steplength):
	S = np.zeros((len(orbitals), len(orbitals)))
	for i, orbital1 in enumerate(orbitals):
		for j, orbital2 in enumerate(orbitals):
			S[i][j] = evalTwoOrbitalFunctionWithContractedGaussians(primativeOverlapIntegral, orbital1, orbital2, steplength)
	return S