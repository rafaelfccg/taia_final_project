import benchmarkFunctions
import numpy
import random

MAX_ITERATIONS = 1000
NUM_DIMENSIONS = 4
POPULATION_SIZE = 50
NUM_TRIALS = 25

USE_CROSS_OVER = True

X_MAX = 32.0
X_MIN = -32.0
Y_MAX = 32.0
Y_MIN = -32.0

EPS = 1e-5

D_MAX = 0.005
N_MAX = 0.01
OMEGA_N = 0.9
OMEGA_F = 0.9
V_f = 0.02
C_t = 0.5


# Global iteration value
curr_iteration = 0

# Global fitness values
Xworst = (list(), 0.0, list(), 0.0)
Xbest = (list(), 1e10, list(), 1e10)

############################################
# Begin helpers
############################################

# Implemented according to PEP 485
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def initialize_function(benchmark_params, dims):
    global fitness
    global X_MIN
    global X_MAX
    global EPS
    global NUM_DIMENSIONS

    fitness = benchmark_params[0]
    if dims==None:
        NUM_DIMENSIONS = benchmark_params[1]
    else:
        NUM_DIMENSIONS = dims
    EPS = benchmark_params[2]
    X_MIN = benchmark_params[3]
    X_MAX = benchmark_params[4]

    if fitness == benchmarkFunctions.branin:
        global Y_MIN
        global Y_MAX
        global generate_population
        Y_MIN = benchmark_params[5]
        Y_MAX = benchmark_params[6]
        generate_population = generate_population_branin

#-------------------------------------------
# End helpers
#-------------------------------------------

############################################
# Begin local alpha calculation functions
############################################

def xij_hat(Xi, Xj):
    distance = numpy.subtract(Xj[0], Xi[0])
    return numpy.divide(distance, numpy.linalg.norm(distance) + EPS)

def kij_hat(Xi, Xj):
    return float(Xi[1] - Xj[1]) / (Xworst[1] - Xbest[1])

def sensing_distance(Xi, population):
    constant = 1.0 / (5.0 * POPULATION_SIZE)
    distance = 0.0
    for Xj in population:
        distance += numpy.linalg.norm(numpy.subtract(Xi[0], Xj[0]))
    return distance * constant

def detect_neighbours(Xi, d, population):
    neighbours = list()
    for Xj in population:
        euc_dist = numpy.linalg.norm(numpy.subtract(Xi[0], Xj[0]))
        if euc_dist < d and not isclose(euc_dist, 0.0):
            neighbours.append(Xj)
    return neighbours

# Each alpha is a P-sized list where each element is an N-dimensional array
def local_alpha(population):
    alphas = list()
    for Xi in population:
        alphai = [0.0 for i in range(NUM_DIMENSIONS)]
        distance = sensing_distance(Xi, population)
        neighbours = detect_neighbours(Xi, distance, population)
        for Xj in neighbours:
            curr = [kij_hat(Xi, Xj) * x for x in xij_hat(Xi, Xj)]
            alphai = numpy.add(alphai, curr)
        alphas.append(alphai)
    return alphas

#-------------------------------------------
# End local alpha calculation functions
#-------------------------------------------

############################################
# Begin target alpha calculation functions #
############################################

# Each alpha is a P-sized list where each element is an N-dimensional array
def target_alpha(population):
    Cbest = 2.0 * (random.uniform(0.0, 1.0) + (float(curr_iteration) / MAX_ITERATIONS))
    alphas = list()
    for Xi in population:
        alphai = [Cbest * kij_hat(Xi, Xbest) * x for x in xij_hat(Xi, Xbest)]
        alphas.append(alphai)
    return alphas

#-------------------------------------------
# End target alpha calculation functions
#-------------------------------------------

############################################
# Begin foraging motion functions
############################################

def find_food(population):
    enum = [0.0 for i in range(NUM_DIMENSIONS)]
    denom = 0.0
    for Xi in population:
        enum = numpy.add(enum, [(1.0 / Xi[1]) * x for x in Xi[0]])
        denom += 1.0 / Xi[1]
    position = numpy.divide(enum, denom)
    return (position, fitness(position))

# Each beta is a P-sized list where each element is an N-dimensional array
def beta_food(population):
    Xfood = find_food(population)
    Cfood = 2.0 * (1.0 - (float(curr_iteration) / MAX_ITERATIONS))
    betas = list()
    for Xi in population:
        betai = [Cfood * kij_hat(Xi, Xfood) * x for x in xij_hat(Xi, Xfood)]
        betas.append(betai)
    return betas

# Each beta is a P-sized list where each element is an N-dimensional array
def beta_best(population):
    betas = list()
    for Xi in population:
        betai = [kij_hat(Xi, Xi[2:]) * x for x in xij_hat(Xi, Xi[2:])]
        betas.append(betai)
    return betas

#-------------------------------------------
# End foraging motion functions
#-------------------------------------------

############################################
# Begin physical diffusion functions
############################################

def Di():
    direction = [random.uniform(-1.0, 1.0) for i in range(NUM_DIMENSIONS)]
    constant = D_MAX * (1.0 - (float(curr_iteration) / MAX_ITERATIONS))
    return [constant * x for x in direction]

#-------------------------------------------
# End physical diffusion functions
#-------------------------------------------

############################################
# Begin motion process functions
############################################

# Each Ni is n P-sized list where each element is an N-dimensional array
def Ni(population, Ni_old):
    N = list()
    alphas = numpy.add(local_alpha(population), target_alpha(population))
    for i in range(len(population)):
        Ni_ = numpy.add([N_MAX * x for x in alphas[i]], 
                        [OMEGA_N * x for x in Ni_old[i]])
        N.append(Ni_)
    return N

# Each Fi is n P-sized list where each element is an N-dimensional array
def Fi(population, Fi_old):
    F = list()
    betas = numpy.add(beta_food(population), beta_best(population))
    #betas = beta_food(population)
    for i in range(len(population)):
        Fi_ = numpy.add([V_f * x for x in betas[i]],
                        [OMEGA_F * x for x in Fi_old[i]])
        F.append(Fi_)
    return F

def dt(population):
    upper_bounds = list()
    lower_bounds = list()
    for i in range(NUM_DIMENSIONS):
        upper = -1e10
        lower = 1e10
        for Xi in population:
            if Xi[0][i] > upper:
                upper = Xi[0][i]
            if Xi[0][i] < lower:
                lower = Xi[0][i]
        upper_bounds.append(upper)
        lower_bounds.append(lower)
    return C_t * sum(numpy.subtract(upper_bounds, lower_bounds))

def move():
    global curr_iteration
    global OMEGA_F
    global OMEGA_N
    global Xworst
    global Xbest

    Xworst = (list(), 0.0, list(), 0.0)
    Xbest = (list(), 1e10, list(), 1e10)
    
    curr_iteration = 0.0

    population = generate_population()
    zero = [0.0 for i in range(NUM_DIMENSIONS)]
    Ni_old = [list(zero) for i in range(POPULATION_SIZE)]
    Fi_old = [list(zero) for i in range(POPULATION_SIZE)]

    for it in range(MAX_ITERATIONS):
        #print "#" + str(it) + " Best: " + str(Xbest[1]) + " Worst: " + str(Xworst[1])

        # Fitness evaluation
        set_fitness_bounds(population)
        set_history_fitness_bounds(population)

        # Calculate motion
        Ni_curr = list(Ni(population, Ni_old))
        Fi_curr = list(Fi(population, Ni_old))
        dX_dt = numpy.add(Ni_curr, Fi_curr)
        dX_dt = numpy.add(dX_dt, [Di() for n in range(POPULATION_SIZE)])
        Ni_old = list(Ni_curr)
        Fi_old = list(Fi_curr)

        # Update positions
        new_population = list()
        dt_ = dt(population)
        for i in range(POPULATION_SIZE):
            step = [dt_ * x for x in dX_dt[i]]
            Xi_new = list(population[i])
            Xi_new[0] = numpy.add(Xi_new[0], step)
            Xi_new[1] = fitness(Xi_new[0])
            new_population.append(tuple(Xi_new))

        if USE_CROSS_OVER:
            cross_over_operador(new_population)

        population = list(new_population)

        # Linearly decrease inertia
        OMEGA_F = 0.9 - ((curr_iteration * 0.8) / MAX_ITERATIONS)
        OMEGA_N = 0.9 - ((curr_iteration * 0.8) / MAX_ITERATIONS)

        # Go to next iteration
        curr_iteration += 1
    return (population, Xbest)


#-------------------------------------------
# End motion process functions
#-------------------------------------------

############################################
# Begin evolutionary functions
############################################

# Change the benchmark function in the main() call in order to modify the fitness evaluation
fitness = benchmarkFunctions.ackley

# Each genome follows this pattern:
# (<n-dimensional position>, <fitness>, <best position in history>, <best fitness in history>)
def generate_population():
    population = list()
    for i in range(POPULATION_SIZE):
        genome = list()
        for s in range(NUM_DIMENSIONS):
            coord = random.uniform(X_MIN, X_MAX);
            genome.append(coord)
        population.append((genome, fitness(genome), genome, fitness(genome)))
    set_fitness_bounds(population)
    set_history_fitness_bounds(population)
    return population

def generate_population_branin():
    population = list()
    for i in range(POPULATION_SIZE):
        genome = list()
        coord1 = random.uniform(X_MIN, X_MAX)
        coord2 = random.uniform(Y_MIN, Y_MAX);
        genome.extend([coord1,coord2])
        population.append((genome, fitness(genome), genome, fitness(genome)))
    set_fitness_bounds(population)
    set_history_fitness_bounds(population)
    return population

def set_fitness_bounds(population):
    global Xworst
    global Xbest
    for Xi in population:
        if Xi[1] < Xbest[1]:
            Xbest = Xi
        if Xi[1] > Xworst[1]:
            Xworst = Xi

def set_history_fitness_bounds(population):
    for Xi in population:
        Xi_ = list(Xi)
        # If current fitness is better than the best in history, swap
        if Xi[1] < Xi[3]:
            Xi_[3] = Xi[1]
            Xi_[2] = list(Xi[0])
        Xi = tuple(Xi_)

def cross_over_operador(population):
    pop_len = len(population)
    for Xi in population:
        Cr = 0.2 * kij_hat(Xi, Xbest)
        for d in range(NUM_DIMENSIONS):
            if random.uniform(0,1) < Cr:
                j = random.randint(0,pop_len - 1)
                Xj = population[j]
                Xi[0][d] = Xj[0][d]

#-------------------------------------------
# End evolutionary functions
#-------------------------------------------

def main(function_params, dims = None):
    initialize_function(function_params, dims)
    best = 1e10
    avg = 0.0
    std = 0.0
    for i in range(NUM_TRIALS):
        print "Running trial #" + str(i+1)
        (p, b) = move()
        best = min(b[1], best)
        avg += numpy.mean([x[1] for x in p])
        std += numpy.std([x[1] for x in p])
    avg = avg / NUM_TRIALS
    std = std / NUM_TRIALS
    print "Best out of " + str(NUM_TRIALS) + " runs: " + str(best)
    print "Average out of " + str(NUM_TRIALS) + " runs: " + str(avg)
    print "Std. dev out of " + str(NUM_TRIALS) + " runs: " + str(std)

def test_case_2(benchmark_params):
    dimensions = [2,4,6,8]

    for dim in dimensions:
        print 'DIMENSIONS: ' + str(dim)
        main(benchmark_params, dim)

print "ACKLEY"
test_case_2(benchmarkFunctions.ACKLEY())