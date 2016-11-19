import benchmarkFunctions
import numpy
import random

NUM_DIMENSIONS = 2
POPULATION_SIZE = 10
X_MAX = 32
X_MIN = -32
EPS = 1e-9

# Global fitness values
kworst = 1.0
kbest = 0.1

# Begin helpers

# Implemented acoording to PEP 485
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# End helpers

# Begin local alpha calculation functions
def xij_hat(Xi, Xj):
    distance = numpy.subtract(Xj[0], Xi[0])
    return numpy.divide(distance, numpy.linalg.norm(distance) + EPS)

def kij_hat(Xi, Xj):
    return float(Xi[1] - Xj[1]) / (kworst - kbest)

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
    return alphas

# End local alpha calculation functions

# Begin target alpha calculation functions


# End target alpha calculation functions

# Begin evolutionary functions

fitness = benchmarkFunctions.ackley

def generate_population():
    population = list()
    for i in range(POPULATION_SIZE):
        genome = list()
        for s in range(NUM_DIMENSIONS):
            coord = random.uniform(X_MIN, X_MAX);
            genome.append(coord)
        population.append((genome, fitness(genome)))
    return population

# End evolutionary functions

pop = generate_population()
a = local_alpha(pop)
print a
