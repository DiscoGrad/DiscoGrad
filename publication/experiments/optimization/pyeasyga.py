# -*- coding: utf-8 -*-
"""
    pyeasyga module

    source: https://raw.githubusercontent.com/remiomosowon/pyeasyga/develop/pyeasyga/pyeasyga.py
"""

import random
import copy
from concurrent import futures
from operator import attrgetter

from six.moves import range

import numpy as np

class GeneticAlgorithm(object):
    """Genetic Algorithm class.

    This is the main class that controls the functionality of the Genetic
    Algorithm.

    A simple example of usage:

    >>> # Select only two items from the list and maximise profit
    >>> from pyeasyga.pyeasyga import GeneticAlgorithm
    >>> input_data = [('pear', 50), ('apple', 35), ('banana', 40)]
    >>> easyga = GeneticAlgorithm(input_data)
    >>> def fitness (member, data):
    >>>     return sum([profit for (selected, (fruit, profit)) in
    >>>                 zip(member, data) if selected and
    >>>                 member.count(1) == 2])
    >>> easyga.fitness_function = fitness
    >>> easyga.run()
    >>> print easyga.best_individual()

    """

    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 elitism=True,
                 maximise_fitness=True,
                 verbose=False,
                 random_state=None,
                 param_init_func=None):
        """Instantiate the Genetic Algorithm.

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation
        :param int: random seed. defaults to None

        """

        self.seed_data = seed_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness
        self.verbose = verbose

        self.random = random.Random(random_state)

        self.current_generation = []

        self.tournament_size = self.population_size // 10

        self.random_state = np.random.RandomState()

        assert(param_init_func != None)
        self.param_init_func = param_init_func

    def create_individual(self, seed_data):
        """Create a candidate solution representation.

        e.g. for a bit array representation:

        >>> return [random.randint(0, 1) for _ in range(len(data))]

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :returns: candidate solution representation as a list

        """
        #return [self.random.random() for _ in range(len(seed_data))]
        return list(self.param_init_func(self.random_state))

    def crossover(self, parent_1, parent_2):
        """Crossover (mate) two parents to produce two children.

        :param parent_1: candidate solution representation (list)
        :param parent_2: candidate solution representation (list)
        :returns: tuple containing two children

        """
        index = self.random.randrange(1, len(parent_1)+1)
        child_1 = parent_1[:index] + parent_2[index:]
        child_2 = parent_2[:index] + parent_1[index:]
        return child_1, child_2

    def mutate(self, individual):
        """Reverse the bit of a random index in an individual."""
        mutate_index = self.random.randrange(len(individual))
        #individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]
        individual[mutate_index] = self.param_init_func(self.random_state)[0]

    def random_selection(self, population):
        """Select and return a random member of the population."""
        return self.random.choice(population)

    def tournament_selection(self, population):
        """Select a random number of individuals from the population and
        return the fittest member of them all.
        """
        if self.tournament_size == 0:
            self.tournament_size = 2
        members = self.random.sample(population, self.tournament_size)
        members.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)
        return members[0]


    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual(self.seed_data)
            individual = Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self, n_workers=None, parallel_type="processing"):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        # If using a single worker, run on a simple for loop to avoid losing
        # time creating processes.
        if n_workers == 1:
            fitnesses = self.fitness_function([individual.genes for individual in self.current_generation], self.seed_data)
            for i, individual in enumerate(self.current_generation):
              individual.fitness = fitnesses[i]
            #for individual in self.current_generation:
            #    individual.fitness = self.fitness_function(
            #        individual.genes, self.seed_data)
        else:

            if "process" in parallel_type.lower():
                executor = futures.ProcessPoolExecutor(max_workers=n_workers)
            else:
                executor = futures.ThreadPoolExecutor(max_workers=n_workers)

            # Create two lists from the same size to be passed as args to the
            # map function.
            genes = [individual.genes for individual in self.current_generation]
            eata = [self.seed_data for _ in self.current_generation]

            with executor as pool:
                results = pool.map(self.fitness_function, genes, self.seed_data)

            for individual, result in zip(self.current_generation, results):
                individual.fitness = result

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.tournament_selection

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = self.random.random() < self.crossover_probability
            can_mutate = self.random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate(child_1.genes)
                self.mutate(child_2.genes)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    def create_first_generation(self, n_workers=None, parallel_type="processing"):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness(
            n_workers=n_workers, parallel_type=parallel_type
        )
        self.rank_population()

    def create_next_generation(self, n_workers=None, parallel_type="processing"):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        self.calculate_population_fitness(
            n_workers=n_workers, parallel_type=parallel_type
        )
        self.rank_population()
        if self.verbose:
            print("Fitness: %f" % self.best_individual()[0])

    # note: added by justinnk
    def initialize(self):
        self.create_first_generation(
            n_workers=1, parallel_type=""
        )

    # note: added by justinnk
    def step(self):
        self.create_next_generation(
            n_workers=1, parallel_type=""
        )

    def run(self, n_workers=None, parallel_type="processing"):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation(
                n_workers=n_workers, parallel_type=parallel_type
            )

        for _ in range(1, self.generations):
            self.create_next_generation(
                n_workers=n_workers, parallel_type=parallel_type
            )

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return (best.fitness, best.genes)

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.genes) for member
                in self.current_generation)


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fitness = 0

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))

