import random
import math
import benchmarkFunctions

NUM_DIMENSIONS = 20


NUM_ITERATIONS = 1000
POPULATION_SIZE = 100

random_range_value = 10

INERTIA_NEIGHBORS = 0.9
INERTIA_FOOD = 0.9
CT = 0.5

N_MAX = 0.01
FORAGING_SPEED = 0.02
DIFUSION_SPEED = 0.010

EPSILON = 1e-3

X_MAX = 32
X_MIN = -32

fitness = benchmarkFunctions.ackley
kbest = 10**9
kworst = 0

SOLUTION_FOUND_ITERATIONS = list()
CONVERGENT_EXECS = 0
CONVERGENT_INDIVIDUALS = list()
ALL_SOLVED_ITERATIONS = list()
INDIVIDUALS_FITNESS = list()

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

def make_rand_vector(dims):
    vec = [random.uniform(-random_range_value, random_range_value) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [float(x) / mag for x in vec]

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
	return vector_constant_product(make_rand_vector(NUM_DIMENSIONS), DIFUSION_SPEED * (1.0 - float(iteration / NUM_ITERATIONS)))

def distance(v1, v2):
	return norm(vector_diff(v1,v2))

def k_hat(ki, kj):
	global kworst
	global kbest
	return float(ki - kj) / (kworst - kbest)

def x_hat(xi, xj):
	diff = vector_diff(xj,xi)
	norm_diff = norm(diff)
	return [float(x) / (norm_diff + 1e-9) for x in diff]

def alfa_local(krill, krill_fit, population, population_fitness):
	(neighbors, neighbors_fit) = find_neighbors(krill, population, population_fitness)
	# print "num neighbors:" +str(len(neighbors))
	# print "neighbors:" +str(neighbors)
	sum_vec = zero_vector(NUM_DIMENSIONS)
	for idx, value in enumerate(neighbors):
		sum_vec = vector_sum(sum_vec, k_x_hat_product(krill, value, krill_fit, neighbors_fit[idx]))

	return sum_vec

def find_neighbors(krill, population, population_fitness):
	ds = 0.9 * sensing_distance(krill,population)
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
	return (1.0 / (1.0 * (POPULATION_SIZE))) * sum(map(lambda x : distance(krill, x[0]), population))

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
	return 2 * (random.uniform(0,1) + (float(iteration) / NUM_ITERATIONS))

def food_position(population, population_fitness):
	sum_denominator = 0
	sum_numerator = zero_vector(len(population[0][0]))
	for idx, krill in enumerate(population):
		fit_weight = 1.0 /population_fitness[idx]
		sum_numerator = vector_sum(sum_numerator, vector_constant_product(krill[0],fit_weight))
		sum_denominator += fit_weight

	# print sum_numerator
	# print sum_denominator

	return vector_constant_product(sum_numerator, 1.0 / sum_denominator)

def beta_food(krill, krill_fit, food_pos, iteration):
	# print (food_pos)
	food_fit = fitness(food_pos)
	return  vector_constant_product(k_x_hat_product(krill, food_pos, krill_fit, food_fit), C_food(iteration))

def C_food(iteration):
	return 2 * (1.0 - (float(iteration) / NUM_ITERATIONS))

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
	return krill + delta_t * delta_move

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

# def delta_t():
# 	nv = NUM_DIMENSIONS
# 	sumi = 0 
# 	for i in range(NUM_DIMENSIONS):
# 		sumi = sumi + X_MAX - X_MIN

# 	return CT*sumi

def delta_t(population):
    nv = NUM_DIMENSIONS
    sumi = 0 
    lower_bound = list(population[0][0])
    upper_bound = list(population[0][0])

    for x in population:
        for xi in range(NUM_DIMENSIONS):
            if lower_bound[xi] > x[0][xi]:
                lower_bound[xi] = x[0][xi]

            if upper_bound[xi] < x[0][xi]:
                upper_bound[xi] = x[0][xi]

    #print str(lower_bound) + " " + str(upper_bound)

    for x in range(NUM_DIMENSIONS):
        sumi += upper_bound[x] - lower_bound[x]

    return CT * sumi

def check_for_solution(population):
	solutions = 0
	for x in population:
		if abs(fitness(x[0])) < EPSILON :
			solutions += 1

	return solutions

def evolve():
    global CONVERGENT_EXECS
    global kworst
    global kbest

    population = generate_population()
    solved = False
    # print population
    for i in range(NUM_ITERATIONS):
            
        (best_krill, population_fitness) = select_best_krill(population)
        x_food = food_position(population, population_fitness)
        new_population = list()
        iteration_min_fit = min(population_fitness)
        iteration_max_fit = max(population_fitness)
       	if kworst < iteration_max_fit:
       		kworst = iteration_max_fit 

       	if kbest > iteration_min_fit: 
       		kbest =  iteration_min_fit

       	print "iteration "+ str(i)+ ": kworst = "+ str(kworst)+ " | kbest = "+ str(kbest)
        
		# print population
        if i % 10 == 0 :
        	print population_fitness 

        for idx, krill in enumerate(population):
        	krill_best = krill[1]
        	(movement_vector, new_N, new_F) = dX_dt(krill[0], population_fitness[idx], krill_best, best_krill[0], x_food ,population, population_fitness, krill[2], krill[3],i)
        	new_krill_position = vector_sum(krill[0] ,vector_constant_product(movement_vector ,delta_t(population)))
        	
        	# print movement_vector 

        	if fitness(new_krill_position) < fitness(krill_best): 
        		krill_best =  new_krill_position

        	new_population.append((new_krill_position, krill_best, new_N, new_F));

        # if USE_RECOMBINATION:
        #     offspring = generate_offspring(population)

        solutions = check_for_solution(new_population)
        population = new_population

        if solutions > 0 and not solved:
            solved = True
            CONVERGENT_INDIVIDUALS.append(solutions)
            SOLUTION_FOUND_ITERATIONS.append(i)
            #mean_pop_fitness = mean(map(lambda x: ackley(x), population))
            #INDIVIDUALS_FITNESS.append(mean_pop_fitness)
            CONVERGENT_EXECS+=1
            print "Solution found after " + str(i) + " iterations"
            #print "Population fitness: " + str(mean_pop_fitness)
            print "Convergent individuals: " + str(solutions)
            return
        elif solutions==100:
            ALL_SOLVED_ITERATIONS.append(i)
            print "All individuals converged at iter " + str(i)
            return

    print "No solution found after " + str(NUM_ITERATIONS) + " iterations"


evolve()


