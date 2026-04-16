import numpy as np
from new_guassian_tools import derivativeOfTwoGaussianFunc

def evalTwoOrbitalFunctionWithHermiteGaussians(func, hermiteGaussian1, hermiteGaussian2, steplength):
	return derivativeOfTwoGaussianFunc(func, hermiteGaussian1, hermiteGaussian2, hermiteGaussian1.xyzDerivativeOrders, hermiteGaussian2.xyzDerivativeOrders, steplength)

def evalTwoOrbitalFunctionWithCartesianGaussians(func, cartesianGaussian1, cartesianGaussian2, steplength):
	xCoeffs1, yCoeffs1, zCoeffs1 = hermiteCoeffsFromCartesianGaussian(cartesianGaussian1)
	xCoeffs2, yCoeffs2, zCoeffs2 = hermiteCoeffsFromCartesianGaussian(cartesianGaussian2)
	print(xCoeffs1, yCoeffs1, zCoeffs1)
	print(xCoeffs2, yCoeffs2, zCoeffs2)
	total = 0
	for x1order, x1coeff in enumerate(xCoeffs1):
		for y1order, y1coeff in enumerate(yCoeffs1):
			for z1order, z1coeff in enumerate(zCoeffs1):
				for x2order, x2coeff in enumerate(xCoeffs2):
					for y2order, y2coeff in enumerate(yCoeffs2):
						for z2order, z2coeff in enumerate(zCoeffs2):
							if x1coeff == 0 or y1coeff == 0 or z1coeff == 0 or x2coeff == 0 or y2coeff == 0 or z2coeff == 0:
								continue
							print("---")
							print(x1order, y1order, z1order, x2order, y2order, z2order)
							print(x1coeff, y1coeff, z1coeff, x2coeff, y2coeff, z2coeff)
							hermiteGaussian1 = HermiteGaussian(cartesianGaussian1.pos, cartesianGaussian1.exponent, np.array([x1order, y1order, z1order]))
							overallCoeff1 = x1coeff * y1coeff * z1coeff
							hermiteGaussian2 = HermiteGaussian(cartesianGaussian2.pos, cartesianGaussian2.exponent, np.array([x2order, y2order, z2order]))
							overallCoeff2 = x2coeff * y2coeff * z2coeff
							print(overallCoeff1 * overallCoeff2, evalTwoOrbitalFunctionWithHermiteGaussians(func, hermiteGaussian1, hermiteGaussian2, steplength), overallCoeff1 * overallCoeff2 * evalTwoOrbitalFunctionWithHermiteGaussians(func, hermiteGaussian1, hermiteGaussian2, steplength))
							total += overallCoeff1 * overallCoeff2 * evalTwoOrbitalFunctionWithHermiteGaussians(func, hermiteGaussian1, hermiteGaussian2, steplength)
	return total


class HermiteGaussian:
	def __init__(self, pos, exponent, xyzDerivativeOrders):
		self.pos = pos
		self.exponent = exponent
		self.xyzDerivativeOrders = xyzDerivativeOrders

class CartesianGaussian:
	def __init__(self, pos, exponent, xyzPowers):
		self.pos = pos
		self.exponent = exponent
		self.xyzPowers = xyzPowers

class ContractedGaussian:
	def __init__(self, cartesianGaussians, coefficients):
		self.cartesianGaussians = cartesianGaussians
		self.coefficients = coefficients


# Matrix where you multiply it with a coefficient vector of gaussian derivatives, and it returns the vector of coeffs for cartesian gaussians. So the inverse does the opposite, cartesian to derivatives.
def hermiteCoeffMatrix(order, exponent):
	num_rows = order
	coeffMatrix = np.zeros((num_rows, num_rows))
	coeffMatrix[0][0] = 1
	for row in range(1, num_rows):
		tmpRow = coeffMatrix[row - 1]
		indicies = np.arange(num_rows)
		multply_x_roll_mask = np.ones((num_rows,))
		multply_x_roll_mask[0] = 0
		derive_roll_mask = np.ones((num_rows,))
		derive_roll_mask[-1] = 0
		tmpRow = -2 * np.roll(tmpRow, 1) * exponent * multply_x_roll_mask + np.roll(tmpRow * indicies, -1) * derive_roll_mask
		coeffMatrix[row] = tmpRow
	return np.transpose(coeffMatrix)


def hermiteCoeffsFromCartesianGaussian(cartesianGaussian):
	maxPower = np.max(cartesianGaussian.xyzPowers)
	mat = hermiteCoeffMatrix(maxPower + 1, cartesianGaussian.exponent)
	matInv = np.linalg.inv(mat)
	xPower = np.zeros(maxPower + 1)
	xPower[cartesianGaussian.xyzPowers[0]] = 1
	yPower = np.zeros(maxPower + 1)
	yPower[cartesianGaussian.xyzPowers[1]] = 1
	zPower = np.zeros(maxPower + 1)
	zPower[cartesianGaussian.xyzPowers[2]] = 1
	xCoeffs = matInv @ xPower
	yCoeffs = matInv @ yPower
	zCoeffs = matInv @ zPower
	return xCoeffs, yCoeffs, zCoeffs