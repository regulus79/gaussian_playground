
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

orbitals = []
nuclei = []
for i in range(len(atom_positions)):
	orbitals.append(ContractedGaussian([CartesianGaussian(atom_positions[i], exponent, np.array([0,0,0]))], [1]))
	nuclei.append(Nucleus(atom_positions[i], atom_charges[i]))

eigenvalues, eigenvectors = orbtialEigs(nuclei, orbitals, 0.01)
print("Calculated MO coeffs")

print("Eigenvalues (eV):", np.sort(eigenvalues) / charge_e)
print("Eigenvectors:")
for i in np.argsort(eigenvalues):
	print(np.round(eigenvectors[:, i], 4))

plotMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, num_cols=4)