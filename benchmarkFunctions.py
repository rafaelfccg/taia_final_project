import math
import sys
import operator

FUNCTION_EVALUATION = 0
EPS = 1e-5

def product(list):
	p = 1
	for i in list:
		p *= i
	return p

ackley_max_x = 32
ackley_min_x = -32

griewank_max_x = 600
griewank_min_x = -600

rastrigin_max_x = 5.12
rastrigin_min_x = -5.12

rosenbrock_max_x = 30
rosenbrock_min_x = -30

schwefel226_max_x = 500
schwefel226_min_x = -500

schwefel222_max_x = 100
schwefel222_min_x = -100

sphere_max_x = 100
sphere_min_x = -100

# 20 dimensions
# minimum 0
def ackley(genome):
	global FUNCTION_EVALUATION
	FUNCTION_EVALUATION += 1
	sum1 = sum(map(lambda x : x ** 2, genome))
	sum2 = sum(map(lambda x : math.cos(2 * math.pi * x), genome))
	n = len(genome)
	a = -0.2 * math.sqrt(sum1/n)
	b = (sum2/n)
	return -20 * math.exp(a) - math.exp(b) + 20 + math.e

def ACKLEY():
	return [ackley, 20, EPS, ackley_min_x, ackley_max_x]

# minimum 0
def griewank (genome):
	n = len(genome)
	sumi = 1 + (sum(map(lambda x : x ** 2, genome)))/4000
	prod_vector = [math.cos(genome[i]/math.sqrt(i+1)) for i in range(n)]
	prod = reduce(operator.mul, prod_vector, 1)

	return sumi - prod

def GRIEWANK():
	return [griewank, 20, EPS, griewank_min_x, griewank_max_x]

# minimum 0
def rastrigin (genome):
	n = len(genome)
	sumi = 10 * n + sum(map(lambda x : x**2 - 10 * math.cos(2 * math.pi * x), genome))
	return sumi

def RASTRIGIN():
	return [rastrigin, 20, EPS, rastrigin_min_x, rastrigin_max_x]

# minimum 0
def rosenbrock(genome):
	n = len(genome)
	sumi = sum([100 * (genome[i+1] - genome[i]**2)**2 + (genome[i] - 1)**2 for i in range(n-1)])
	return sumi

def ROSENBROCK():
	return [rosenbrock, 20, EPS, rosenbrock_min_x, rosenbrock_max_x]

# minimum 0
def Schwefel226(genome):
	return (sum(map(lambda x : -x * math.sin(math.sqrt(math.fabs(x))), genome)))

def SCHWEFEL226():
	return [Schwefel226, 20, EPS, schwefel226_min_x, schwefel226_max_x]

def Schwefel222(genome):
	modified_genome = map(lambda x : math.fabs(x), genome)
	return sum(modified_genome) + product(modified_genome)

def SCHWEFEL222():
	return [Schwefel222, 20, EPS, schwefel222_min_x, schwefel222_max_x]

def Sphere(genome):
	return math.sqrt(sum(map(lambda x : x**2, genome)))

def SPHERE():
	return [Sphere, 20, EPS, sphere_min_x, sphere_max_x]

# 2 dimensions

branin_max_x = 10
branin_min_x = -5

branin_max_y = 15
branin_min_y = 0

#minimum 5/4*pi
def branin(genome):
	b = 1.25/(math.pi ** 2)
	c = 5/math.pi
	d = 6
	g = 10
	h = 0.125/math.pi

	return ((genome[1] - b * (genome[0]**2) + c * genome[0] - d)**2) + g *(1 - h) * math.cos(genome[0]) + g

def BRANIN():
	return [branin, 2, 5/(4*math.pi), branin_min_x, branin_max_x, branin_min_y, branin_max_y]

alpine_min_x = -10
alpine_max_x = 10

#minimum 0
def alpine(genome):
	return sum(map(lambda xi: math.fabs(xi*math.sin(xi)+0.1*xi), genome))

def ALPINE():
	return [alpine, 4, EPS, alpine_min_x, alpine_max_x]

# # dimensions 4
# PS: Ta faltando valores na tabela de hartman1 que ele menciona no artigo
# hartman1_max_x = 1
# hartman1_min_x = 0

# # minimum -3.86
# def hartman1(genome):
# 	ci = [1,1.2,3,3.2]
# 	aij = [[3,10,30],[0.1, 10, 35],[3,10,30],[0.1,10,35]]
# 	pij = [[0.3689,0.117,0.2673],[0.4699, 0.4387, 0.747],[0.1091,0.8732,0.5547],[0.03815, 0.5743, 0.8828]]
# 	for x in xrange(1,10):
# 		pass
# 	return