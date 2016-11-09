import math
import sys

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

# 20 dimensions
# minimum 0
def ackley(genome):
    n = len(genome)
    sum1 = sum(map(lambda x : x ** 2, genome))
    sum2 = sum(map(lambda x : math.cos(2 * math.pi * x), genome))
    a = -0.2 * math.sqrt((1.0 / n) * sum1)
    b = (1.0 / n) * sum2
    return -20 * math.exp(a) - math.exp(b) + 20 + math.e

# minimum 0
def griewank (genome):
	n = len(genome)
	sumi = 1 + (sum(map(lambda x : x ** 2, genome)))/4000
	prod = 1;
	for i in range(n):
		prod = math.cos(genome[i]/i)

	return sumi - prod

# minimum 0
def rastrigin (genome):
	n = len(genome)
	sumi = 10 * n + sum(map(lambda x : x**2 - 10 * math.cos(2 * math.pi * x), genome))
	return sumi

# minimum 0
def rosenbrock(genome):
	n = len(genome)
	sumi = 0
	for i in range(n-1):
		sumi += 100 * (genome[i+1] - genome[i]**2)**2 + (genome[i] - 1)**2
	return sumi

# minimum 0
def Schwefel226(genome):
	return (sum(map(lambda x : -x * math.sin(math.sqrt(math.fabs(x))), genome)))

# 2 dimensions

branin_max_x = 10
branin_min_x = -5

branin_max_y = 15
branin_min_y = 0

#minimum 5/4*pi
def branin(genome):
	a = 1
	b = 1.25/(math.pi ** 2)
	c = 5/math.pi
	d = 6
	g = 10
	h = 0.125/math.pi

	return a * ((genome[1] - b * (genome[0]**2) + c * genome[0] - d)**2) * g *(1 - h) * math.cos(genome[0]) + g



print (Schwefel226([1,1]))