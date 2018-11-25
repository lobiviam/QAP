import random
import issue
import numpy as np


class GeneticAlg:
    def __init__(self, issue):
        self.issue = issue
        self.initial_population = InitialPopulation(50, self.issue)

    def crossover(self, parents):
        pass

    def selection(self, population, n_childrens):
        probs = []
        fitness_sum = sum(1/chromosome.objective_function for chromosome in population.chromosomes)
        for i in range(len(population.chromosomes)):
            chromosome = population.chromosomes[i]
            prob = (1/chromosome.objective_function)/fitness_sum
            probs.append(prob)
        childrens_index_list = np.random.choice(len(population.chromosomes), n_childrens, probs)
        childrens_list = []
        for i in range(len(childrens_index_list)):
            childrens_list.append(population.chromosomes[childrens_index_list[i]])
        return childrens_list

class Population:
    def __init__(self, issue, chromosomes):
        self.issue = issue
        self.chromosomes = chromosomes


class InitialPopulation:
    def __init__(self, init_popul_dim, _issue):
        self.dimension = _issue.dimension
        self._issue = _issue
        self.chromosomes = self.get_initial_population(init_popul_dim)

    def get_initial_population(self, init_popul_dim):
        init_population_list = []
        init_gene = list(range(self.dimension))
        for i in range(init_popul_dim):
            random.shuffle(init_gene)
            chromosome = Chromosome(init_gene, self._issue)
            init_population_list.append(chromosome)
        return init_population_list


class Chromosome:
    def __init__(self, genes_list, issue):
        self.genes_list = genes_list
        self.objective_function = self.get_objective_function(issue)
        print(self.objective_function)

    def get_objective_function(self, issue):
        obj_func_value = 0
        for i in range(issue.dimension):
            for j in range(issue.dimension):
                obj_func_value += issue.distance_matrix[i, j] * issue.flow_matrix[self.genes_list[i]][
                    self.genes_list[j]]
        return obj_func_value
