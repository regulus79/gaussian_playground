
# Molecular Orbital Calculations

This script uses Gaussian orbitals to calculate the approximate eigenstates of the single-electron Hamiltonian of a molecule, given the positions of all the atoms. This script does not take into account any electron-electron interactions.

Running the `calculate_eigenstates.py` script and passing in an xyz file will calculate the coefficients to approximate the eigenstates of the system from a linear combination of Gaussian orbitals. By default, 3 sphereically-symmetric gaussians of varying exponents are used for hydrogen, and larger atoms such as carbon have an addition 9 p-like orbitals, which are simply gaussians multiplied by x, y, or z to give a dumbbell shape, similar to an atomic p orbital.

The computed eigenstates can be plotted in 3d space, along with the corresponding eigenvalues give the energy of that state. Once the calculations are finished, the script uses matplotlib to display a surface/contour of the 3d electron density at a specified quantile level (by default 0.9).

# Examples

Several example molecules are provided in `example_xyz_files/*.xyz` which store the x,y,z positions of each atom. This info is parsed by the script to generate the potential landscape of the molecule based on the positive charges of all the nuclei.

### Carbon Dioxide

Carbon dioxide (CO2) is a linear molecule with two double bonds between the atoms: O=C=O.

```
python calculate_eigenstates.py example_xyz_files/carbon_dioxide.xyz --quantile 0.99 --plot occupied
```

<img width="1472" height="806" alt="image" src="https://github.com/user-attachments/assets/45efa76b-6838-4159-b2e8-1ee200929d65" />

### Water

Water (H2O) is a bent molecule, with an electronegative oxygen atom singly-bonded to two hydrogen atoms.

```
python calculate_eigenstates.py example_xyz_files/water.xyz --quantile 0.80 --plot occupied --extra_unoccipied 2
```

<img width="1530" height="819" alt="image" src="https://github.com/user-attachments/assets/fba14071-69c5-44bc-8569-a77ff0f190f6" />


## Usage

```
python calculate_eigenstates.py <input_file.xyz>

Optional:
	--quantile 0.9
	--plot [occupied or homo_lumo]
	--num_frontier 2
	--extra_unoccipied 0
	--orbital_multiplicity 3
	--lattice_size 40
	--buffer 1.0
```

By default, the highest occupied molecular orbital (HOMO) and the lowest unoccipied molecular orbital (LUMO) are plotted, along with 3 extra orbitals above and below. This can be changed with `--num_frontier n`, which will show n occupied and unoccipied orbitals above and below.
Additionally, `--plot occupied` will show all occupied orbitals. In this mode, `--extra_unoccipied` controls how many higher energy, unoccipied orbitals will be shown in addition to the occupied.

By default, 3 gaussians per orbital shape are used with different exponents, to provide a better basis set for approximating the true shape of the eigenstates. This number can be changed with `--orbital_multiplicity`.

If matplotlib is taking too long to plot the orbitals, you can decrease the resolution by changing `--lattice_size` to something smaller. This is the size of the 3d lattice which is used when evaluating the wavefunction in 3d space so that it can be plotted. This is not used when catually calculating the eigenstates; that is done all analytically (well, almost. The integrals of p orbitals are currently computed using a finite difference method to take the derivative of the integral of a corresponding spherical gaussian. This can be done analytically but I have not yet implemented the equations.)

`--buffer` helps with plotting. If the orbitals are too big that they extend out of view, this can be increased to give more room around the atoms for the 3d plot.

