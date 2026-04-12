import math
import numpy as np


# Includes alternating +1/-1 sign since this is supposed to be used as a raw derivative of a gaussian
# so it might not match sign with the wikipedia page with physicist's hermite polynomials
# exponent is b and argument is x for derivative og e^-(bx^2)
def hermitePolynomial(order, exponent, argument):
	if order <= 0:
		return 1
	coeffs = np.zeros((order+1,))
	coeffs[0] = 1
	indicies = np.arange(order+1)
	multply_x_roll_mask = np.ones((order+1,))
	multply_x_roll_mask[0] = 0
	derive_roll_mask = np.ones((order+1,))
	derive_roll_mask[-1] = 0
	for i in range(order):
		coeffs = -2 * np.roll(coeffs, 1) * exponent * multply_x_roll_mask + np.roll(coeffs * indicies, -1) * derive_roll_mask
	return np.sum(coeffs * argument**indicies)


class GaussianOrbital():
	def __init__(self, xderivativePower, yderivativePower, zderivativePower, xpos, ypos, zpos, exponent):
		self.xderivativePower = xderivativePower
		self.yderivativePower = yderivativePower
		self.zderivativePower = zderivativePower
		self.xpos = xpos
		self.ypos = ypos
		self.zpos = zpos
		self.exponent = exponent



def gaussianProductConstants(pos1, exponent1, pos2, exponent2):
	newExponent = exponent1 + exponent2
	newPos = (exponent1 * pos1 + exponent2 * pos2) / (exponent1 + exponent2)
	scalingConstantExponent = (exponent1 * exponent2) / (exponent1 + exponent2)
	scalingConstantArgument = pos2 - pos1

	return newExponent, newPos, scalingConstantExponent, scalingConstantArgument


def guassianInnerProduct(pos1, exponent1, pos2, exponent2):
	productExponent, productPos, scalingConstantExponent, scalingConstantArgument = gaussianProductConstants(pos1, exponent1, pos2, exponent2)

	scalingFactor = math.exp(-scalingConstantExponent * scalingConstantArgument**2)
	integralOfProductGuassian = scalingFactor * math.sqrt(math.pi / productExponent)

	return integralOfProductGuassian

def integralOfNormalGuassianToInfinity(lowerBound, exponent):
	return (0.5 + 0.5*math.erf(-lowerBound * math.sqrt(exponent))) * math.sqrt(math.pi / exponent)

def coulombDerivativeExpansionCoeff(functionDerivativeOrder, totalDerivatives):
	oneOverXDerivativeOrder = totalDerivatives - functionDerivativeOrder
	coeff = math.factorial(oneOverXDerivativeOrder) * (-1)**oneOverXDerivativeOrder * math.comb(totalDerivatives, functionDerivativeOrder)
	return coeff

def gaussianCoulombIntegral(gaussianPos, exponent, derivativeXPower, derivativeYPower, derivativeZPower, potentialPos):
	totalDerivatives = derivativeXPower + derivativeYPower + derivativeZPower

	offset = gaussianPos - potentialPos
	distance = np.sum(offset**2)**0.5
	
	overallCoeff = math.pi / exponent
	
	integral1 = integralOfNormalGuassianToInfinity(-distance, exponent)
	integral2 = integralOfNormalGuassianToInfinity(distance, exponent)

	# Find the nth derivative with respect to the distance
	overallDerivative = 0
	for derivativeOrder in range(totalDerivatives + 1):
		derivativeExpansionCoeff = coulombDerivativeExpansionCoeff(derivativeOrder, totalDerivatives)
		oneOverDistancePower = totalDerivatives - derivativeOrder + 1
		functionDerivative = None
		if derivativeOrder == 0:
			functionDerivative = integral1 - integral2
		else:
			functionDerivative = hermitePolynomial(derivativeOrder - 1, exponent, distance) * (math.exp(-exponent * distance**2) + math.exp(-exponent * distance**2))
		#print("derivative order", derivativeOrder, "coeff", derivativeExpansionCoeff, "1/x power", oneOverDistancePower, "derivative", functionDerivative)
		overallDerivative += derivativeExpansionCoeff * (1/distance) ** oneOverDistancePower * functionDerivative
	#print(overallCoeff / distance * (integral1 - integral2))
	


	direction = offset / distance
	derivativeDistanceWrtX = np.sum(np.array([1,0,0]) * direction)
	derivativeDistanceWrtY = np.sum(np.array([0,1,0]) * direction)
	derivativeDistanceWrtZ = np.sum(np.array([0,0,1]) * direction)
	#print(derivativeDistanceWrtX, derivativeDistanceWrtY, derivativeDistanceWrtZ)
	#print(overallCoeff * overallDerivative, (derivativeDistanceWrtX**derivativeXPower * derivativeDistanceWrtY**derivativeYPower * derivativeDistanceWrtZ**derivativeZPower))

	return  overallCoeff * overallDerivative * derivativeDistanceWrtX**derivativeXPower * derivativeDistanceWrtY**derivativeYPower * derivativeDistanceWrtZ**derivativeZPower

