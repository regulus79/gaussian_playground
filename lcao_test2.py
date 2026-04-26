
from new_guassian_tools import *
from contracted_gaussian_tools import *
from lcao_tools import *
from visualize_tools import *
import numpy as np


exponent = 0.5 / 1e-10**2

pos1 = np.array([0,0,0]) * 1e-10
pos2 = np.array([1,0,0]) * 1e-10
pos3 = np.array([2,0,0]) * 1e-10
pos4 = np.array([3,0,0]) * 1e-10

cartesianGaussian1 = CartesianGaussian(pos1, exponent, np.array([0,0,1]))
contractedGaussian1 = ContractedGaussian([cartesianGaussian1], [1,1])

cartesianGaussian2 = CartesianGaussian(pos2, exponent, np.array([0,0,1]))
contractedGaussian2 = ContractedGaussian([cartesianGaussian2], [1,1])

cartesianGaussian3 = CartesianGaussian(pos3, exponent, np.array([0,0,1]))
contractedGaussian3 = ContractedGaussian([cartesianGaussian3], [1,1])

cartesianGaussian4 = CartesianGaussian(pos4, exponent, np.array([0,0,1]))
contractedGaussian4 = ContractedGaussian([cartesianGaussian4], [1,1])

orbitals = [contractedGaussian1, contractedGaussian2, contractedGaussian3, contractedGaussian4]
nuclei = [Nucleus(pos1, 1), Nucleus(pos2, 1), Nucleus(pos3, 1), Nucleus(pos4, 1)]

#orb2 = ContractedGaussian([CartesianGaussian(pos1, 0.4 / bohr_radius**2, np.array([1,0,0]))], [1])
eigval, eigvecs = orbtialEigs(nuclei, orbitals, 0.01 * bohr_radius)
print("eigval:",eigval, "in eV:", eigval / charge_e)
print("eigvecs:",eigvecs)
plotMOs(eigval, eigvecs, orbitals, nuclei)
#ax = setupAxes("testing", bohr_radius)
#plotOrbitals([orb], [1], ax)
#plt.show()