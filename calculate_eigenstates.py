
from lcao_tools import *
from visualize_tools import *

import parse_xyz

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("--quantile", type=float, default=0.90)
parser.add_argument("--lattice_size", type=int, default=40)
parser.add_argument("--buffer", type=float, default=1.0)
parser.add_argument("--orbital_multiplicity", type=int, default=3)
parser.add_argument("--plot", type=str, choices=["occupied", "homo_lumo"], default="homo_lumo")
parser.add_argument("--num_frontier", type=int, default=4)
args = parser.parse_args()
print(f"Parsing file {args.inputfile}")

atom_positions = []
atom_charges = []

with open(args.inputfile, "r") as file:
	atom_positions, atom_charges = parse_xyz.parse(file.read())

print("Atom positions (m):", atom_positions)
print("Atom charges:", atom_charges)

# Somewhat arbitrary values for how large the gaussian basis functions should be initially
exponent = 0.2 / bohr_radius**2
# Generate more gaussians with different exponents
exponent_factors = []#[2**0, 2**0.5, 2**1, 2**1.5, 2**2, 2**2.5, 2**3, 2**4, 2**5, 2**6, 2**8, 2**10]
for i in range(args.orbital_multiplicity):
	# Arbitrary spacing of exponents, going 2**0, 2**3, 2**6, 2**9, etc
	exponent_factors.append(2**(3*i))

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
#print("Eigenvectors:")
#for i in np.argsort(eigenvalues):
	#print(eigenvectors[i])
	#print(np.round(eigenvectors[i], 4))
print("Total Occupied Energy (eV):", occupiedElectronEnergy(eigenvalues, nuclei) / charge_e)

#plotMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, lattice_shape=(args.lattice_size, args.lattice_size, args.lattice_size), buffer=args.buffer, num_cols=4)
if args.plot == "occupied":
	plotOccupiedMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, lattice_shape=(args.lattice_size, args.lattice_size, args.lattice_size), buffer=args.buffer, num_cols=4, plus_extra = 0)
elif args.plot == "homo_lumo":
	plotFrontierMOs(eigenvalues, eigenvectors, orbitals, nuclei, quantile=args.quantile, lattice_shape=(args.lattice_size, args.lattice_size, args.lattice_size), buffer=args.buffer, num_cols=4, num_frontier = args.num_frontier)
