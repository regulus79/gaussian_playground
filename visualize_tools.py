import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


lattice_shape = (20, 20, 20)
length_per_step = 3 / 20
lattice_radius = 5


def plotOrbitals(contractedGaussians, coeffs, ax, quantile = 0.5):
	psi = np.zeros(lattice_shape)
	coords_1d = np.linspace(-lattice_radius, lattice_radius, 20)
	xv, yv, zv = np.meshgrid(coords_1d, coords_1d, coords_1d)
	for j, contractedGaussian in enumerate(contractedGaussians):
		contractedCoeff = coeffs[j]
		for i, cartesianGaussian in enumerate(contractedGaussian.cartesianGaussians):
			cartesianCoeff = contractedGaussian.coefficients[i]
			exponent = cartesianGaussian.exponent
			xPower = cartesianGaussian.xyzPowers[0]
			yPower = cartesianGaussian.xyzPowers[1]
			zPower = cartesianGaussian.xyzPowers[2]
			psi += contractedCoeff * cartesianCoeff * np.exp(-exponent * ((xv - cartesianGaussian.pos[0])**2 + (yv - cartesianGaussian.pos[1])**2 + (zv - cartesianGaussian.pos[2])**2)) * ((xv - cartesianGaussian.pos[0])**xPower) * ((yv - cartesianGaussian.pos[1])**yPower) * ((zv - cartesianGaussian.pos[2])**zPower)
	scale = 2 * lattice_radius
	pos = np.array([0,0,0])#contractedGaussian.cartesianGaussians[0].pos # THIS Assumes all gaussians have same center
	plotPsi(psi, scale, pos, ax, quantile)


def plotPsi(psi, scale, pos, ax, quantile = 0.5):
	prob_density = np.absolute(psi)**2
	prob_density /= np.sum(prob_density)
	level = np.quantile(prob_density, 1 - quantile, weights = prob_density, method = "inverted_cdf")
	verts, faces, normals, values = measure.marching_cubes(prob_density, level)
	facecolors = [colors.hsv_to_rgb((np.angle(psi[tuple(np.round(np.mean(verts[face], axis = 0)).astype(int))]) / (2*math.pi) + 0.5, 1, 1)) for face in faces]
	verts -= 0.5 * np.array([psi.shape[0], psi.shape[1], psi.shape[2]])
	verts *= scale / psi.shape[0]
	verts += pos
	mesh = Poly3DCollection(verts[faces], shade=True, facecolors = facecolors)
	ax.add_collection3d(mesh)

def setupAxes(title, plot_radius, rows = 1, cols = 1, index = 1):
	ax = plt.subplot(rows, cols, index, projection = "3d")
	ax.title.set_text(title)
	ax.set_xlim3d(-plot_radius, plot_radius)
	ax.set_ylim3d(-plot_radius, plot_radius)
	ax.set_zlim3d(-plot_radius, plot_radius)
	ax.set_aspect("equal")
	return ax


def plotAtomPositions(ax, nuclei):
	for nucleus in nuclei:
		ax.scatter(nucleus.pos[0], nucleus.pos[1], nucleus.pos[2], color = "black")