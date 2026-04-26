
from lcao_tools import *
from visualize_tools import *

import parse_xyz

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("--quantile", type=float, default=0.5)
args = parser.parse_args()
print(f"Parsing file {args.inputfile}")

atom_positions = []
atom_charges = []

with open(args.inputfile, "r") as file:
	atom_positions, atom_charges = parse_xyz.parse(file.read())

print("Atom positions (m):", atom_positions)
print("Atom charges:", atom_charges)

exponent = 0.5 / bohr_radius**2
exponent_factors = [0.5, 1, 2]

orbitals = []
nuclei = []
for i in range(len(atom_positions)):
	for mult in exponent_factors:
		base_exponent = exponent * mult #* math.sqrt(atom_charges[i]**2) # THIS IS WRONG
		orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent, np.array([0,0,0]))], [1]))
		if atom_charges[i] > 1:
			orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent / 4, np.array([1,0,0]))], [1]))
			orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent / 4, np.array([0,1,0]))], [1]))
			orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], base_exponent / 4, np.array([0,0,1]))], [1]))
	nuclei.append(Nucleus(atom_positions[i], atom_charges[i]))

print(f"{len(orbitals)} orbitals, {len(nuclei)} nuclei")

eigenvalues, eigenvectors = orbtialEigs(nuclei, orbitals, 0.01)
print("Calculated MO coeffs")

print("Eigenvalues (eV):", np.sort(eigenvalues) / charge_e)
print("Eigenvectors:")
for i in np.argsort(eigenvalues):
	print(np.round(eigenvectors[:, i], 4))

#plotMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, num_cols=4)
plotOccupiedMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, num_cols=4, plus_extra = 2)