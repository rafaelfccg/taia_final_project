import benchmarkFunctions
import numpy
import random

MAX_ITERATIONS = 1000
NUM_DIMENSIONS = 2
POPULATION_SIZE = 10

X_MAX = 32
X_MIN = -32

EPS = 1e-9

D_MAX = 0.002

# Global iteration value
curr_iteration = 1

# Global fitness values
Xworst = (list(), 0.0, list(), 0.0)
Xbest = (list(), 1e10, list(), 1e10)

############################################
# Begin helpers
############################################

# Implemented acoording to PEP 485
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

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
    constant = 1.0 / (1.0 * POPULATION_SIZE)
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

def local_alpha(population):
    alphas = list()
    for Xi in population:
        #print Xi
        alphai = [0.0 for i in range(NUM_DIMENSIONS)]
        distance = sensing_distance(Xi, population)
        #print "distance " + str(distance)
        neighbours = detect_neighbours(Xi, distance, population)
        #print "neighbours "+ str(neighbours)
        for Xj in neighbours:
            curr = [kij_hat(Xi, Xj) * x for x in xij_hat(Xi, Xj)]
            # print xij_hat(Xi, Xj)
            # print kij_hat(Xi, Xj)
            # print curr
            alphai = numpy.add(alphai, curr)
        alphas.append(alphai)
    #print alphas
    return alphas

#-------------------------------------------
# End local alpha calculation functions
#-------------------------------------------

############################################
# Begin target alpha calculation functions #
############################################

def target_alpha(population):
    Cbest = 2.0 * (random.uniform(0.0, 1.0) + (curr_iteration / MAX_ITERATIONS))
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

def beta_food(population):
    Xfood = find_food(population)
    Cfood = 2.0 * (1.0 - (curr_iteration / MAX_ITERATIONS))
    betas = list()
    for Xi in population:
        betai = [Cfood * kij_hat(Xi, Xfood) * x for x in xij_hat(Xi, Xfood)]
        betas.append(betai)
    return betas

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
    constant = D_MAX * (1.0 - (curr_iteration / MAX_ITERATIONS))
    return [constant * x for x in direction]

#-------------------------------------------
# End physical diffusion functions
#-------------------------------------------

############################################
# Begin motion process functions
############################################

# TODO continue here with Fi, Ni, Di and Dx/dt

#-------------------------------------------
# End motion process functions
#-------------------------------------------

############################################
# Begin evolutionary functions
############################################

# Change the benchmark function here in order to modify the fitness evaluation
fitness = benchmarkFunctions.ackley

def generate_population():
    population = list()
    for i in range(POPULATION_SIZE):
        genome = list()
        for s in range(NUM_DIMENSIONS):
            coord = random.uniform(X_MIN, X_MAX);
            genome.append(coord)
        population.append((genome, fitness(genome), genome, fitness(genome)))
    set_fitness_bounds(population)
    set_history_bounds(population)
    return population

def set_fitness_bounds(population):
    global Xworst
    global Xbest
    for Xi in population:
        if Xi[1] < Xbest[1]:
            Xbest = Xi
        if Xi[1] > Xworst[1]:
            Xworst = Xi

def set_history_bounds(population):
    global Xworst_history
    global Xbest_history
    for Xi in population:
        # If current fitness is better than the best in history, swap
        if Xi[1] < Xi[3]:
            Xi[3] = Xi[1]
            Xi[2] = list(Xi[0])

#-------------------------------------------
# End evolutionary functions
#-------------------------------------------


# Debug
pop = generate_population()
# a = local_alpha(pop)
# b = target_alpha(pop)
# beta_best(pop)
# print Di()