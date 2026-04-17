
from new_guassian_tools import *
from contracted_gaussian_tools import *
from lcao_tools import *
from visualize_tools import *
import numpy as np



pos1 = np.array([0,0,0])
pos2 = np.array([2,0,0])
pos3 = np.array([4,1,2])

cartesianGaussian1 = CartesianGaussian(pos1, 1, np.array([0,0,0]))
contractedGaussian1 = ContractedGaussian([cartesianGaussian1], [1,1])

cartesianGaussian2 = CartesianGaussian(pos2, 1, np.array([0,0,0]))
contractedGaussian2 = ContractedGaussian([cartesianGaussian2], [1,1])

cartesianGaussian3 = CartesianGaussian(pos3, 1, np.array([0,0,0]))
contractedGaussian3 = ContractedGaussian([cartesianGaussian3], [1,1])

orbitals = [contractedGaussian1, contractedGaussian2, contractedGaussian3]
nuclei = [Nucleus(pos1, 1), Nucleus(pos2, 1), Nucleus(pos3, 1)]

orb = ContractedGaussian([CartesianGaussian(pos1, 0.5 / bohr_radius**2, np.array([0,0,0]))], [1])
eigval, eigvecs = orbtialEigs([Nucleus(pos1, 1)], [orb], 0.01 * bohr_radius)
print("eigval:",eigval, "in eV:", eigval / charge_e)
#ax = setupAxes("testing", bohr_radius)
#plotOrbitals([orb], [1], ax)
#plt.show()
exit()

eigenvalues, eigenvectors = orbtialEigs(nuclei, orbitals, 0.01)
print(eigenvalues, eigenvectors)

plotMOs(eigenvalues, eigenvectors, orbitals, nuclei)

#ax = setupAxes("testing", 4)
#plotAtomPositions(ax, nuclei)
#plotOrbitals([contractedGaussian1], [1], ax)
#plotOrbitals([contractedGaussian2], [1], ax)
#plotOrbitals([contractedGaussian3], [1], ax)
#plotOrbitals([contractedGaussian1, contractedGaussian2], [1,1,1], ax)

#plotOrbitals([contractedGaussian1, contractedGaussian2, contractedGaussian3], [1,1,1], ax)
#plt.show()

exit()

#orbital1 = PrimativeGaussian(np.array([0,0,0])*1.0, 1)
#orbital2 = PrimativeGaussian(np.array([2,2,2])*1.0, 1)

# x^2 * y^2
# 2xy^2
# 4xy
#def testFunc(orb1, orb2):
#	return 3*orb1.pos[0] * orb2.pos[0]

#print(testFunc(orbital1, orbital2))

#print(derivativeOfTwoGaussianFunc(testFunc, orbital1, orbital2, np.array([1,0,0]), np.array([0,0,0]), 0.001))
#print(derivativeOfTwoGaussianFunc(primativeOverlapIntegral, orbital1, orbital2, np.array([2,1,0]), np.array([2,1,0]), 0.01))

#exit()

#cartesianGaussian1 = CartesianGaussian(np.array([0,0,0]), 1.5, np.array([2,0,0]))
#cartesianGaussian2 = CartesianGaussian(np.array([2,2,2]), 3, np.array([2,0,0]))

#print(evalTwoOrbitalFunctionWithCartesianGaussians(primativeOverlapIntegral, cartesianGaussian1, cartesianGaussian2, 0.01))

#exit()


#cartesianGaussian1 = CartesianGaussian(np.array([0,0,0]), 0.5, np.array([0,0,0]))
#cartesianGaussian2 = CartesianGaussian(np.array([0,0,0]), 4, np.array([0,0,0]))
#contractedGaussian1 = ContractedGaussian([cartesianGaussian1, cartesianGaussian2], [1,1])

#cartesianGaussian3 = CartesianGaussian(np.array([2,2,2]), 1, np.array([0,0,0]))
#cartesianGaussian4 = CartesianGaussian(np.array([2,2,2]), 3, np.array([0,0,0]))
#contractedGaussian2 = ContractedGaussian([cartesianGaussian3, cartesianGaussian4], [1,1])

#print(evalTwoOrbitalFunctionWithContractedGaussians(primativeOverlapIntegral, contractedGaussian1, contractedGaussian2, 0.01)) # NICE

#exit()


#exit()


#print(primativeOverlapIntegral(orbital1, orbital2))

#print(derivativeOfTwoGaussianFunc(primativeOverlapIntegral, orbital1, orbital2, np.array([2,0,0]), np.array([2,0,0]), 0.01))

#print(primativeCoulombIntegral(orbital1, orbital2, potentialPos))
#print(derivativeOfTwoGaussianFunc(primativeCoulombIntegral, orbital1, orbital2, np.array([1,0,0]), np.array([1,0,0]), 0.001, potentialPos))

#(V^T V)^-1 * V^T H V x = L x

#print(primativeOverlapIntegral(orbital1, orbital1))
#print(primativeKineticIntegral(orbital1, orbital1, 0.01))


steplength = 0.01
potentialMagnitude = -1
offset = 1

orbital1 = PrimativeGaussian(np.array([0,0,0])*1.0, 1)
orbital2 = PrimativeGaussian(np.array([offset,0,0])*1.0, 1)
potentialPos1 = np.array([0,0,0])
potentialPos2 = np.array([offset,0,0])

H = np.zeros((2,2))
H[0,0] = potentialMagnitude * (primativeCoulombIntegral(orbital1, orbital1, potentialPos1) + primativeCoulombIntegral(orbital1, orbital1, potentialPos2)) + -primativeKineticIntegral(orbital1, orbital1, steplength)
H[1,0] = potentialMagnitude * (primativeCoulombIntegral(orbital1, orbital2, potentialPos1) + primativeCoulombIntegral(orbital1, orbital2, potentialPos2)) + -primativeKineticIntegral(orbital1, orbital2, steplength)
H[0,1] = potentialMagnitude * (primativeCoulombIntegral(orbital2, orbital1, potentialPos1) + primativeCoulombIntegral(orbital2, orbital1, potentialPos2)) + -primativeKineticIntegral(orbital2, orbital1, steplength)
H[1,1] = potentialMagnitude * (primativeCoulombIntegral(orbital2, orbital2, potentialPos1) + primativeCoulombIntegral(orbital2, orbital2, potentialPos2)) + -primativeKineticIntegral(orbital2, orbital2, steplength)

S = np.zeros((2,2))
S[0,0] = primativeOverlapIntegral(orbital1, orbital1)
S[1,0] = primativeOverlapIntegral(orbital1, orbital2)
S[0,1] = primativeOverlapIntegral(orbital2, orbital1)
S[1,1] = primativeOverlapIntegral(orbital2, orbital2)

print(H)
print(S)

A = np.linalg.inv(S) @ H
print(A)

print(np.linalg.eig(A))