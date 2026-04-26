
from lcao_tools import *
from visualize_tools import *

import parse_xyz

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("--quantile", type=float, default=0.5)
parser.add_argument("--lattice_size", type=int, default=20)
parser.add_argument("--buffer", type=float, default=1.0)
args = parser.parse_args()
print(f"Parsing file {args.inputfile}")

atom_positions = []
atom_charges = []

with open(args.inputfile, "r") as file:
	atom_positions, atom_charges = parse_xyz.parse(file.read())

print("Atom positions (m):", atom_positions)
print("Atom charges:", atom_charges)

exponent = 0.1 / bohr_radius**2
exponent_factors = [2**0, 2**0.5, 2**1, 2**1.5, 2**2, 2**2.5, 2**3, 2**4, 2**5, 2**6, 2**8, 2**10]#[2**0, 2**3, 2**6, 2**9]#[2**1,2**2,2**3,2**4,2**6,2**8]

orbitals = []
nuclei = []
for i in range(len(atom_positions)):
	for mult in exponent_factors:
		base_exponent = exponent * mult #* math.sqrt(atom_charges[i]**2) # THIS IS WRONG
		orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent, np.array([0,0,0]))], [1]))
		if atom_charges[i] > 1:
			orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent, np.array([1,0,0]))], [1]))
			orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent, np.array([0,1,0]))], [1]))
			orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent, np.array([0,0,1]))], [1]))
	nuclei.append(Nucleus(atom_positions[i], atom_charges[i]))

print(f"{len(orbitals)} orbitals, {len(nuclei)} nuclei")

eigenvalues, eigenvectors = orbtialEigs(nuclei, orbitals, 0.01 * bohr_radius)
print("Calculated MO coeffs")

print("Eigenvalues (eV):", np.sort(eigenvalues) / charge_e)
print("Eigenvectors:")
for i in np.argsort(eigenvalues):
	print(eigenvectors[i])
	#print(np.round(eigenvectors[i], 4))
print("Total Occupied Energy (eV):", occupiedElectronEnergy(eigenvalues, nuclei) / charge_e)

#plotMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, lattice_shape=(args.lattice_size, args.lattice_size, args.lattice_size), buffer=args.buffer, num_cols=4)
plotOccupiedMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, lattice_shape=(args.lattice_size, args.lattice_size, args.lattice_size), buffer=args.buffer, num_cols=4, plus_extra = 2)