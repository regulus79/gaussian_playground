import math
import numpy as np

bohr_radius = 5.291772e-11
h_bar = 6.626070e-34 / (2*math.pi)
mass_e = 9.109384e-31
charge_e = 1.602176e-19
epsilon0 = 8.854188e-12

ALMOST_ZERO = 1e-20


def derivativeOfTwoGaussianFunc(func, orbital1, orbital2, derivatives1, derivatives2, steplength, *args):
	# Sign flips since we are differentiating wrt to the gaussian centers, not x technically
	signFromX1Derivative = (-1)**derivatives1[0]
	signFromY1Derivative = (-1)**derivatives1[1]
	signFromZ1Derivative = (-1)**derivatives1[2]
	signFromX2Derivative = (-1)**derivatives2[0]
	signFromY2Derivative = (-1)**derivatives2[1]
	signFromZ2Derivative = (-1)**derivatives2[2]
	totalSign = signFromX1Derivative * signFromY1Derivative * signFromZ1Derivative * signFromX2Derivative * signFromY2Derivative * signFromZ2Derivative
	totalDerivative = 0
	for x1 in range(derivatives1[0] + 1):
		for y1 in range(derivatives1[1] + 1):
			for z1 in range(derivatives1[2] + 1):
				for x2 in range(derivatives2[0] + 1):
					for y2 in range(derivatives2[1] + 1):
						for z2 in range(derivatives2[2] + 1):
							offset1 = (np.array([x1,y1,z1]) - derivatives1 / 2) * steplength
							offset2 = (np.array([x2,y2,z2]) - derivatives2 / 2) * steplength
							tmpOrbital1 = PrimativeGaussian(orbital1.pos*1.0, orbital1.exponent)
							tmpOrbital2 = PrimativeGaussian(orbital2.pos*1.0, orbital2.exponent)
							tmpOrbital1.pos += offset1
							tmpOrbital2.pos += offset2
							coeffx1 = math.comb(derivatives1[0], x1) * (-1)**(derivatives1[0] - x1) / steplength**derivatives1[0]
							coeffy1 = math.comb(derivatives1[1], y1) * (-1)**(derivatives1[1] - y1) / steplength**derivatives1[1]
							coeffz1 = math.comb(derivatives1[2], z1) * (-1)**(derivatives1[2] - z1) / steplength**derivatives1[2]
							coeffx2 = math.comb(derivatives2[0], x2) * (-1)**(derivatives2[0] - x2) / steplength**derivatives2[0]
							coeffy2 = math.comb(derivatives2[1], y2) * (-1)**(derivatives2[1] - y2) / steplength**derivatives2[1]
							coeffz2 = math.comb(derivatives2[2], z2) * (-1)**(derivatives2[2] - z2) / steplength**derivatives2[2]
							#print("---", x1, y1, z1, x2, y2, z2)
							#print(tmpOrbital1.pos, tmpOrbital2.pos)
							#print(coeffx1, coeffy1, coeffz1, coeffx2, coeffy2, coeffz2)
							#print(func(tmpOrbital1, tmpOrbital2, *args))
							totalDerivative += coeffx1 * coeffy1 * coeffz1 * coeffx2 * coeffy2 * coeffz2 * func(tmpOrbital1, tmpOrbital2, *args)
	return totalDerivative * totalSign





class PrimativeGaussian():
	def __init__(self, pos, exponent):
		self.pos = pos
		self.exponent = exponent



def gaussianProductConstants(pos1, exponent1, pos2, exponent2):
	newExponent = exponent1 + exponent2
	newPos = (exponent1 * pos1 + exponent2 * pos2) / (exponent1 + exponent2)
	scalingConstantExponent = (exponent1 * exponent2) / (exponent1 + exponent2)
	scalingConstantArgument = pos2 - pos1
	return newExponent, newPos, scalingConstantExponent, scalingConstantArgument

def primativeGuassianInnerProduct(pos1, exponent1, pos2, exponent2):
	productExponent, productPos, scalingConstantExponent, scalingConstantArgument = gaussianProductConstants(pos1, exponent1, pos2, exponent2)
	scalingFactor = math.exp(-scalingConstantExponent * scalingConstantArgument**2)
	integralOfProductGuassian = scalingFactor * math.sqrt(math.pi / productExponent)
	return integralOfProductGuassian


def primativeOverlapIntegral(orbital1, orbital2):
	overlapX = primativeGuassianInnerProduct(orbital1.pos[0], orbital1.exponent, orbital2.pos[0], orbital2.exponent)
	overlapY = primativeGuassianInnerProduct(orbital1.pos[1], orbital1.exponent, orbital2.pos[1], orbital2.exponent)
	overlapZ = primativeGuassianInnerProduct(orbital1.pos[2], orbital1.exponent, orbital2.pos[2], orbital2.exponent)
	return overlapX * overlapY * overlapZ

# How to take Second derivative of gaussian, inner producted with other gaussian
# -2(b1)*exp(-(b1)(x-c1)^2)*exp(-(b2)(x-c2)^2)  +  4(b1)^2*(x-c1)^2*exp(-(b1)(x-c1)^2)*exp(-(b2)(x-c2)^2)
# -2(b1)*exp(-(b1)(x-c1)^2 - (b2)(x-c2)^2)  +  4(b1)^2*(x-c1)^2*exp(-(b1)(x-c1)^2 - (b2)(x-c2)^2)
# -2(b1)*exp(-(b1)(x-c1)^2 - (b2)(x-c2)^2)  +  4(b1)^2*(x-c1)^2*exp(-(b1*b2)/(b1+b2) * (c1-c2)^2))*exp(-(b1+b2)*(x-(b1c1 + b2c2)/(b1+b2))^2)
# let p = new pos = (b1c1 + b2c2)/(b1+b2)
# -2(b1)*exp(-scalingExponent * scalingArgument^2)*exp(-newExponent * (x - newPos)^2)
#  +  4(b1)^2*(x-c1)^2*exp(-scalingExponent * scalingArgument^2)*exp(-newExponent * (x - newPos)^2)
# The first term equals -2(b1) * overlap = -2(b1) * exp(-scalingExponent * scalingArgument^2) * sqrt(pi/newExponent)
# Now the second term
# 4(b1)^2*(x-c1)^2*exp(-scalingExponent * scalingArgument^2)*exp(-newExponent * (x - newPos)^2)
# since (x-c1)^2 = (x-newPos)^2 + 2*(newPos - c1)(x-newPos) + (newPos - c1)^2
# =   4(b1)^2 * (x-newPos)^2 * exp(-scalingExponent * scalingArgument^2)*exp(-newExponent * (x - newPos)^2)
#   + 4(b1)^2 * 2*(newPos - c1)(x-newPos) * exp(-scalingExponent * scalingArgument^2)*exp(-newExponent * (x - newPos)^2)
#   + 4(b1)^2 * (newPos - c1)^2 * exp(-scalingExponent * scalingArgument^2)*exp(-newExponent * (x - newPos)^2)
# integral of First term is just 4(b1)^2 * 1/(2*newExponent) * sqrt(pi/newExponent) * exp(-scalingExponent * scalingArgument^2)
# integral of Second term is 0
# integral of Third term is 4(b1)^2 * (newPos - c1)^2 * sqrt(pi/newExponent) * exp(-scalingExponent * scalingArgument^2)
# So in total we have
# =   -2(b1) * exp(-scalingExponent * scalingArgument^2) * sqrt(pi/newExponent)
#   + 4(b1)^2 * 1/(2*newExponent) * sqrt(pi/newExponent) * exp(-scalingExponent * scalingArgument^2)
#   + 4(b1)^2 * (newPos - c1)^2 * sqrt(pi/newExponent) * exp(-scalingExponent * scalingArgument^2)
# = exp(-scalingExponent * scalingArgument^2) * sqrt(pi/newExponent) * (-2b1 + 4(b1)^2 * (1/(2*newExponent) + (newPos - c1)^2))
# = overlap * (-2b1 + 4(b1)^2 * (1/(2*newExponent) + (newPos - c1)^2))

def primativeGaussianLaplacianInnerProduct(pos1, exponent1, pos2, exponent2):
	overlap = primativeGuassianInnerProduct(pos1, exponent1, pos2, exponent2)
	newExponent = exponent1 + exponent2
	newPos = (exponent1 * pos1 + exponent2 * pos2) / (exponent1 + exponent2)
	laplacian = overlap * (-2*exponent1 + 4*exponent1**2 * (1/(2*newExponent) + (newPos - pos1)**2))
	return laplacian

def primativeKineticIntegral(orbital1, orbital2, steplength):
	newExponent = orbital1.exponent + orbital2.exponent
	overlapX = primativeGuassianInnerProduct(orbital1.pos[0], orbital1.exponent, orbital2.pos[0], orbital2.exponent)
	overlapY = primativeGuassianInnerProduct(orbital1.pos[1], orbital1.exponent, orbital2.pos[1], orbital2.exponent)
	overlapZ = primativeGuassianInnerProduct(orbital1.pos[2], orbital1.exponent, orbital2.pos[2], orbital2.exponent)
	newPosX = (orbital1.exponent * orbital1.pos[0] + orbital2.exponent * orbital2.pos[0]) / (orbital1.exponent + orbital2.exponent)
	newPosY = (orbital1.exponent * orbital1.pos[1] + orbital2.exponent * orbital2.pos[1]) / (orbital1.exponent + orbital2.exponent)
	newPosZ = (orbital1.exponent * orbital1.pos[2] + orbital2.exponent * orbital2.pos[2]) / (orbital1.exponent + orbital2.exponent)
	laplacianX = overlapX * overlapY * overlapZ * (-2*orbital1.exponent + 4*orbital1.exponent**2 * (1/(2*newExponent) + (newPosX - orbital1.pos[0])**2))
	laplacianY = overlapX * overlapY * overlapZ * (-2*orbital1.exponent + 4*orbital1.exponent**2 * (1/(2*newExponent) + (newPosY - orbital1.pos[1])**2))
	laplacianZ = overlapX * overlapY * overlapZ * (-2*orbital1.exponent + 4*orbital1.exponent**2 * (1/(2*newExponent) + (newPosZ - orbital1.pos[2])**2))
	return -(laplacianX + laplacianY + laplacianZ) * h_bar**2 / (2 * mass_e)


def integralOfNormalGuassianToInfinity(lowerBound, exponent):
	return (0.5 + 0.5*math.erf(-lowerBound * math.sqrt(exponent))) * math.sqrt(math.pi / exponent)

def primativeCoulombIntegral(orbital1, orbital2, potentialPos):
	productExponentX, productPosX, scalingConstantExponentX, scalingConstantArgumentX = gaussianProductConstants(orbital1.pos[0], orbital1.exponent, orbital2.pos[0], orbital2.exponent)
	productExponentY, productPosY, scalingConstantExponentY, scalingConstantArgumentY = gaussianProductConstants(orbital1.pos[1], orbital1.exponent, orbital2.pos[1], orbital2.exponent)
	productExponentZ, productPosZ, scalingConstantExponentZ, scalingConstantArgumentZ = gaussianProductConstants(orbital1.pos[2], orbital1.exponent, orbital2.pos[2], orbital2.exponent)
	scalingFactorX = math.exp(-scalingConstantExponentX * scalingConstantArgumentX**2)
	scalingFactorY = math.exp(-scalingConstantExponentY * scalingConstantArgumentY**2)
	scalingFactorZ = math.exp(-scalingConstantExponentZ * scalingConstantArgumentZ**2)

	exponent = productExponentX
	assert productExponentX == productExponentY and productExponentY == productExponentZ

	productPos = np.array([productPosX, productPosY, productPosZ])
	offset = productPos - potentialPos
	distance = np.sum(offset**2)**0.5

	unitsCoeff = charge_e**2 / (4*math.pi*epsilon0)
	mathCoeff = math.pi / exponent * scalingFactorX * scalingFactorY * scalingFactorZ

	if distance <= ALMOST_ZERO:
		return 2 * mathCoeff * unitsCoeff
	
	integral1 = integralOfNormalGuassianToInfinity(-distance, exponent)
	integral2 = integralOfNormalGuassianToInfinity(distance, exponent)

	return unitsCoeff * mathCoeff * (integral1 - integral2) / distance
