import random
import math

NUM_DIMENSIONS = 20
DIFUSION_SPEED = 0.01

NUM_ITERATIONS = 10000
POPULATION_SIZE = 3

random_range_value = 10

EPSILON = 0.00001

def make_rand_vector(dims):
    vec = [random.uniform(-random_range_value, random_range_value) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def norm(vector):
	return math.sqrt(sum(map(lambda x : x**2, vector)))

def vector_diff(vector1, vector2):
	return [x_i - x_j for x_i, x_j in zip(vector1, vector2)]

def vector_constant_product(vector1, constant):
	return [x_i * constant for x_i in vector1]

def random_difusion(iteration):
	return DIFUSION_SPEED * (1 - iteration/NUM_ITERATIONS) * make_rand_vector(NUM_DIMENSIONS)

def k_hat(ki,kj,kbest,kworst):
	return (ki - kj) / (kworst - kbest)

def x_hat(xi,xj,):
	diff = vector_diff(xj,xi)
	norm_diff = norm(diff)
	return [x/(norm_diff + EPSILON) for x in diff]

def sensing_distance(krill, population):
	return 1.0/(5*POPULATION_SIZE) * sum(map(lambda x : norm(vector_diff(krill,x)), population))

def alfa_local(krill, neighbors):
	return sum(map(lambda x : vector_constant_product(x_hat(krill, x), k_hat(krill, x)), neighbors))

def alfa_target(krill, best, iteration):
	return C_best(iteration) * vector_constant_product(x_hat(krill,best),k_hat(krill,best))

def C_best(iteration):
	return 2 * (random.uniform(0,1) + iteration/NUM_ITERATIONS)


