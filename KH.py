import random
import math

NUM_DIMENSIONS = 20
DIFUSION_SPEED = 0.01

NUM_ITERATIONS = 10000
POPULATION_SIZE = 3

random_range_value = 10

INERTIA_NEIGHBORS = 0.2
INERTIA_FOOD = 0.2
N_MAX = 1
FORAGING_SPEED = 0.02

EPSILON = 0.00001

def make_rand_vector(dims):
    vec = [random.uniform(-random_range_value, random_range_value) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def zero_vector(dims):
	return [0 for i in range(dim)]

def norm(vector):
	return math.sqrt(sum(map(lambda x : x**2, vector)))

def vector_diff(vector1, vector2):
	return [x_i - x_j for x_i, x_j in zip(vector1, vector2)]

def vector_sum(vector1, vector2):
	return [x_i + x_j for x_i, x_j in zip(vector1, vector2)]

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

def alfa_local(krill, population):
	neighbors = find_neighbors(krill)
	return sum(map(lambda x : vector_constant_product(x_hat(krill, x), k_hat(krill, x)), neighbors))

def find_neighbors(krill, population):
	ds = sensing_distance(krill,population)
	pass

def sensing_distance(krill, population):
	return 1.0/(5*POPULATION_SIZE) * sum(map(lambda x : norm(vector_diff(krill,x)), population))

def alfa_target(krill, best, iteration):
	return C_best(iteration) * k_x_hat_product(krill,best)

def k_x_hat_product(krill_i,krill_j):
	return vector_constant_product(x_hat(krill_i,krill_j),k_hat(krill_i,krill_j))

def C_best(iteration):
	return 2 * (random.uniform(0,1) + iteration/NUM_ITERATIONS)

def fitness(krill):
	return 0

def food_position(population):
	sum_denominator = 0
	sum_numerator = zero_vector(len(population[0]))
	for krill in population:
		fit_weight = 1/fitness(krill)
		sum_numerator = vector_sum(sum_numerator, vector_constant_product(krill,fit_weight))
		sum_denominator += fit_weight

	return vector_constant_product(sum_numerator, sum_denominator)

def beta_food(krill,food_position):
	return C_food(iteration) * k_x_hat_product(krill,food_position)

def C_food(iteration):
	return 2*(1 - iteration/NUM_ITERATIONS)

def alfa(krill, best, population):
	return alfa_local(krill_i,population) + alfa_target(krill,best)

def neighbors_induced_mov(krill, best, population, old_N):
	return N_MAX * alfa(krill, best, population) + INERTIA_NEIGHBORS * old_N

def beta(krill, krill_best, population):
	return beta_food(krill, population) + k_x_hat_product(krill, krill_best)

def food_induced_mov(krill, krill_best, population, old_F):
	x_food = food_position(population)
	return FORAGING_SPEED * beta_food(krill) + INERTIA_FOOD * old_F

def dX_dt(krill, krill_best, best,population, old_N, old_F):
	return neighbors_induced_mov(krill, best, population, old_N) +food_induced_mov(krill, krill_best, population,old_F) + random_difusion()

def move(krill, delta_t, delta_move):
	return krill + delta_t * delta_move






	


