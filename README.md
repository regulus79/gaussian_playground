
# Molecular Orbital Calculations

This script uses Gaussian orbitals to calculate the approximate eigenstates of the single-electron Hamiltonian of a molecule, given the positions of all the atoms. This script does not take into account any electron-electron interactions.

# Examples

Several example molecules are provided in `example_xyz_files/*.xyz` which store the x,y,z positions of each atom. This info is parsed by the script to generate the potential landscape of the molecule based on the positive charges of all the nuclei.

### Carbon Dioxide

Carbon dioxide (CO2) is a linear molecule with two double bonds between the atoms: O=C=O. It has 22 electrons in its neutral state (8 + 6 + 8), which can be paired up into 11

```
python calculate_eigenstates.py example_xyz_files/carbon_dioxide.xyz --quantile 0.99 --plot occupied
```

<img width="1472" height="806" alt="image" src="https://github.com/user-attachments/assets/45efa76b-6838-4159-b2e8-1ee200929d65" />

