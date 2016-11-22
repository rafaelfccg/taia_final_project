import random
import math
import benchmarkFunctions
import copy

#sem operadores geneticos

NUM_DIMENSIONS = 20

NUM_ITERATIONS = 1000
POPULATION_SIZE = 50

random_range_value = 1

INERTIA_NEIGHBORS = 0.9
INERTIA_FOOD = 0.9
CT = 0.5

N_MAX = 0.02
FORAGING_SPEED = 0.02
DIFUSION_SPEED = 0.005

EPSILON = 10**-5
CONVERGENCE_PRECISION = 10**-3

X_MAX = 32
X_MIN = -32
Y_MAX = 32
Y_MIN = -32

fitness = benchmarkFunctions.ackley
kbest = 10**9
kworst = 0

SOLUTION_FOUND_ITERATIONS = list()
CONVERGENT_EXECS = 0
CONVERGENT_INDIVIDUALS = list()
ALL_SOLVED_ITERATIONS = list()
INDIVIDUALS_FITNESS = list()
KBEST_FITNESS = list()

# individual representation, (self, self_historical_best, old_N, old_F)
def generate_population():
	population = list()
	for i in range(POPULATION_SIZE):
		genome = list()
		for s in range(NUM_DIMENSIONS):
			individual = random.uniform(X_MIN, X_MAX);
			genome.append(individual)
		population.append((genome, genome, zero_vector(NUM_DIMENSIONS), zero_vector(NUM_DIMENSIONS)))
	return population

def generate_population_branin():
	population = list()
	for i in range(POPULATION_SIZE):
		genome = list()
		individual1 = random.uniform(X_MIN, X_MAX);
		individual2 = random.uniform(Y_MIN, Y_MAX);
		genome.extend([individual1, individual2])
		population.append((genome, genome, zero_vector(NUM_DIMENSIONS), zero_vector(NUM_DIMENSIONS)))
	return population

def make_rand_vector(dims):
	vec = [random.uniform(-random_range_value, random_range_value) for i in range(dims)]
	#mag = sum(x**2 for x in vec) ** .5
	return [x for x in vec]

def zero_vector(dims):
	return [0 for i in range(dims)]

def norm(vector):
	return math.sqrt(sum(map(lambda x : x**2, vector)))

def vector_diff(vector1, vector2):
	return [x_i - x_j for x_i, x_j in zip(vector1, vector2)]

def vector_sum(vector1, vector2):
	return [x_i + x_j for x_i, x_j in zip(vector1, vector2)]

def vector_constant_product(vector1, constant):
	return [x_i * constant for x_i in vector1]

def random_difusion(iteration):
	return vector_constant_product(make_rand_vector(NUM_DIMENSIONS), DIFUSION_SPEED * (1 - iteration/float(NUM_ITERATIONS)))

def distance(v1, v2):
	return norm(vector_diff(v1,v2))

def k_hat(ki, kj):
	return (ki - kj) / (kworst - kbest)

def x_hat(xi, xj):
	diff = vector_diff(xj,xi)
	norm_diff = norm(diff)
	return [x/(norm_diff + EPSILON) for x in diff]

def alfa_local(krill, krill_fit, population, population_fitness):
	(neighbors, neighbors_fit) = find_neighbors(krill, population, population_fitness)
	# print "num neighbors:" +str(len(neighbors))
	# print "neighbors:" +str(neighbors)
	sum_vec = zero_vector(NUM_DIMENSIONS)
	for idx, value in enumerate(neighbors):
		sum_vec = vector_sum(sum_vec, k_x_hat_product(krill, value, krill_fit, neighbors_fit[idx]))

	return sum_vec

def find_neighbors(krill, population, population_fitness):
	ds = sensing_distance(krill,population)
	# print "sensing_distance: " + str(ds)
	neighbors = list()
	neighbors_fit = list()
	for idx, x in enumerate(population):
		individual_i = x[0]
		distance_i = distance(krill,individual_i)
		# print distance_i
		if(individual_i != krill and distance_i <= ds):
			neighbors.append(x[0])
			neighbors_fit.append(population_fitness[idx])

	return (neighbors, neighbors_fit)

def sensing_distance(krill, population):
	val1 = sum(map(lambda x : distance(x[0], krill), population))

	# print val1
	return val1/(POPULATION_SIZE*5)

def alfa_target(krill, krill_fit, best, best_fit, iteration):
	cbest = C_best(iteration)
	return vector_constant_product(k_x_hat_product(krill, best, krill_fit, best_fit),  cbest)

def k_x_hat_product(krill_i,krill_j,fitness_i, fitness_j):
	return vector_constant_product(x_hat(krill_i, krill_j), k_hat(fitness_i, fitness_j))

def alfa(krill, krill_fit, best, population, population_fitness, iteration):
	best_fit = fitness(best)
	local = alfa_local(krill, krill_fit, population, population_fitness)
	target = alfa_target(krill, krill_fit, best, best_fit, iteration)
	# print "local: "+ str(local)
	# print "target: "+ str(target)
	return vector_sum(local,target)

def C_best(iteration):
	return 2 * (random.uniform(0,1) + iteration/float(NUM_ITERATIONS))

def food_position(population, population_fitness):
	sum_denominator = 0
	sum_numerator = zero_vector(len(population[0][0]))
	for idx, krill in enumerate(population):
		fit_weight = 1/population_fitness[idx]
		sum_numerator = vector_sum(sum_numerator, vector_constant_product(krill[0],fit_weight))
		sum_denominator += fit_weight

	return vector_constant_product(sum_numerator, 1/sum_denominator)

def beta_food(krill, krill_fit, food_pos, iteration):
	# print (food_pos)
	food_fit = fitness(food_pos)
	return  vector_constant_product(k_x_hat_product(krill, food_pos, krill_fit, food_fit), C_food(iteration))

def C_food(iteration):
	return 2*(1 - iteration/float(NUM_ITERATIONS))

def neighbors_induced_mov(krill, krill_fit, best, population, population_fitness, old_N, iteration):
	return vector_sum(vector_constant_product(alfa(krill, krill_fit, best, population, population_fitness, iteration), N_MAX), vector_constant_product(old_N, INERTIA_NEIGHBORS))

def beta(krill, krill_fit, krill_best, x_food, population, population_fitness, iteration):
	return vector_sum( beta_food(krill, krill_fit, x_food, iteration), k_x_hat_product(krill, krill_best, krill_fit, fitness(krill_best)))

def food_induced_mov(krill, krill_fit, krill_best, x_food, population, population_fitness, old_F, iteration):
	return vector_sum(vector_constant_product(beta(krill, krill_fit, krill_best, x_food, population, population_fitness, iteration), FORAGING_SPEED), vector_constant_product(old_F, INERTIA_FOOD))

def dX_dt(krill, krill_fit, krill_best, best, x_food, population, population_fitness, old_N, old_F, iteration):
	Ni = neighbors_induced_mov(krill, krill_fit, best, population, population_fitness, old_N, iteration) 
	# print Ni
	Fi = food_induced_mov(krill, krill_fit, krill_best, x_food, population, population_fitness, old_F, iteration) 
	Di = random_difusion(iteration)
	return (vector_sum(vector_sum(Ni,Fi),Di), Ni, Fi)

def move(krill, delta_t, delta_move):
	return vector_sum( krill,vector_constant_product(delta_move, delta_t))

def select_best_krill(population):
	min_krill = population[0]
	min_fitness = 10**9
	population_fitness = list()
	for x in population:
		curr_fit = fitness(x[0])
		population_fitness.append(curr_fit)
		if min_fitness > curr_fit:
			min_krill = x
			min_fitness = curr_fit

	return (min_krill,population_fitness)

def delta_t(population):
	# sumi = 0 
	# lower_bound = copy.copy(population[0][0])
	# upper_bound = copy.copy(population[0][0])

	# for x in population:
	# 	for xi in range(NUM_DIMENSIONS):
	# 		if lower_bound[xi] > x[0][xi]:
	# 			lower_bound[xi] = x[0][xi]

	# 		if upper_bound[xi] < x[0][xi]:
	# 			upper_bound[xi] = x[0][xi]
	 
	meanU = list()

	for x in range(NUM_DIMENSIONS):
		meanU.append(X_MAX-X_MIN)

	# list.sort(meanU)
	# print(meanU)
	return CT *  sum(meanU)

def delta_t_branin(population):
	meanU = list()

	meanU.append(X_MAX-X_MIN)
	meanU.append(Y_MAX-Y_MIN)

	return CT *  sum(meanU)

def check_for_solution(population):
	solutions = 0
	for x in population:
		if abs(fitness(x[1])) < CONVERGENCE_PRECISION :
			solutions += 1

	return solutions

def evolve():
	global CONVERGENT_EXECS
	global kworst
	global kbest
	global INERTIA_NEIGHBORS
	global FORAGING_SPEED

	movement_vector = list()
	population = generate_population()
	krill = population[0]
	solved = False
	
	i = 0
	best_change_iterations = 0
	INERTIA_NEIGHBORS = 0.9
	INERTIA_FOOD = 0.9
	kworst = 0
	kbest = 10**9
	benchmarkFunctions.FUNCTION_EVALUATION = 0
	while i < NUM_ITERATIONS:
		i += 1
		(best_krill, population_fitness) = select_best_krill(population)
		x_food = food_position(population, population_fitness)
		new_population = list()
		iteration_min_fit = min(population_fitness)
		iteration_max_fit = max(population_fitness)
	   	if kworst < iteration_max_fit:
	   		kworst = iteration_max_fit 

	   	if kbest > iteration_min_fit: 
	   		kbest =  iteration_min_fit
	   		best_change_iterations = 0
	   	else:
	   		best_change_iterations += 1

		INERTIA_NEIGHBORS = 0.1 + 0.8 * (1 - i/float(NUM_ITERATIONS))
		INERTIA_FOOD = 0.1 + 0.8 * (1 - i/float(NUM_ITERATIONS))

	   	print "iteration "+ str(i)+ ": kworst = "+ str(kworst)+ " | kbest = "+ str(kbest)
		dt = delta_t(population)
		#print dt
		# print population

		for idx, krill in enumerate(population):
			krill_best = krill[1]
			(movement_vector, new_N, new_F) = dX_dt(krill[0], population_fitness[idx], krill_best, best_krill[0], x_food ,population, population_fitness, krill[2], krill[3],i)
			new_krill_position = vector_sum(krill[0] ,vector_constant_product(movement_vector, dt))

			if fitness(new_krill_position) < fitness(krill_best): 
				krill_best =  new_krill_position

			new_population.append((new_krill_position, krill_best, new_N, new_F));

		# if USE_RECOMBINATION:
		#	 offspring = generate_offspring(population)

		population = new_population

	solutions = check_for_solution(new_population)
	CONVERGENT_INDIVIDUALS.append(solutions)
	SOLUTION_FOUND_ITERATIONS.append(i)
	print SOLUTION_FOUND_ITERATIONS
	kbest_fit = map(lambda x: fitness(x[1]), population)
	mean_pop_fitness = mean(kbest_fit)
	KBEST_FITNESS.append(min(kbest_fit))

	INDIVIDUALS_FITNESS.append(mean_pop_fitness)
	print "best "+ str(population[kbest_fit.index(min(kbest_fit))][1])
	print "Population fitness: " + str(mean_pop_fitness)
	print "Convergent individuals: " + str(solutions)

	if solutions > 0:
		solved = True
		CONVERGENT_EXECS+=1
		print "Solution found after " + str(i) + " iterations"
	else:
		print "No solution found!"

def mean(list_items):
    return sum(list_items)/float(len(list_items))

def std_dev(list_items, mean_items):
    variance_list = map(lambda x : pow(x-mean_items, 2), list_items)
    return math.sqrt(sum(variance_list)/float(len(list_items)))

def initialize_function(benchmark_params):
	global fitness
	global X_MIN
	global X_MAX
	global CONVERGENCE_PRECISION
	global NUM_DIMENSIONS

	fitness = benchmark_params[0]
	NUM_DIMENSIONS = benchmark_params[1]
	CONVERGENCE_PRECISION = benchmark_params[2]
	X_MIN = benchmark_params[3]
	X_MAX = benchmark_params[4]

	is_branin = fitness == benchmarkFunctions.branin
	if is_branin:
		global Y_MIN
		global Y_MAX
		global generate_population
		global delta_t
		Y_MIN = benchmark_params[5]
		Y_MAX = benchmark_params[6]
		generate_population = generate_population_branin
		delta_t = delta_t_branin
	return is_branin

def main(num_of_trials, function_params):
    is_branin = initialize_function(function_params)

    print CONVERGENCE_PRECISION
    print NUM_DIMENSIONS
    print X_MAX
    print X_MIN
    print Y_MAX
    print Y_MIN

    for i in range(num_of_trials):
        print "Execution " + str(i+1)
        evolve()
        print ""

    mean_iterations = mean(SOLUTION_FOUND_ITERATIONS)
    mean_fitness = mean(INDIVIDUALS_FITNESS)
    mean_individuals = mean(CONVERGENT_INDIVIDUALS)
    
    print "Convergent executions: " + str(CONVERGENT_EXECS)
    print "Mean of iterations: " + str(mean_iterations)
    # print "Std of iterations: " + str(std_dev(SOLUTION_FOUND_ITERATIONS, mean_iterations))
    print "Mean of fitness: " + str(mean_fitness)
    print "Std of fitness: " + str(std_dev(INDIVIDUALS_FITNESS, mean_fitness))
    print "Mean of convergent indivs: " + str(mean_individuals)
    print "Std of convergent indivs: " + str(std_dev(CONVERGENT_INDIVIDUALS, mean_individuals))
    print "Best solution found " + str(min(KBEST_FITNESS))
    print "Mean solution found " + str(mean(KBEST_FITNESS))
    # print "Mean of total convergence iterations: " + str(mean_iter_total)
    
#print "ACKLEY"
#main(5, benchmarkFunctions.ACKLEY())
#print "GRIEWANK"
#main(5, benchmarkFunctions.GRIEWANK())
#print "RASTRIGIN"
#main(5, benchmarkFunctions.RASTRIGIN())
print "ROSENBROCK"
main(5, benchmarkFunctions.ROSENBROCK())
print "SCHEWEFEL 226"
main(5, benchmarkFunctions.SCHWEFEL226())
print "SCHEWEFEL 222"
main(5, benchmarkFunctions.SCHWEFEL222())
print "SPHERE"
main(5, benchmarkFunctions.SPHERE())
print "BRANIN"
main(5, benchmarkFunctions.BRANIN())