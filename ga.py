import random

import tqdm

import issue
import numpy as np
from copy import deepcopy


class GeneticAlg:
    def __init__(self, issue):
        self.issue = issue
        self.initial_population = InitialPopulation(100, self.issue)
        self.solution = self.solve()

    def mutation(self, chromosome):
        index1 = random.randint(0, int(len(chromosome.genes_list) / 2))
        index2 = random.randint(int((len(chromosome.genes_list) / 2) + 1), len(chromosome.genes_list) - 1)

        if index1 > index2:
            index1, index2 = index2, index1

        genes_list_copy = deepcopy(chromosome.genes_list)
        genes_list_copy[index1:index2] = np.random.permutation(genes_list_copy[index1:index2])
        return Chromosome(genes_list_copy, self.issue)

    def crossover(self, parents):
        def get_part(index1, index2, parent1_genes):
            return parent1_genes[index1:index2]

        def ordered_crossover(parent1, parent2):
            index1 = random.randint(0, int(len(parent1.genes_list) / 2))
            index2 = random.randint(int((len(parent1.genes_list) / 2) + 1), len(parent1.genes_list) - 1)
            if index1 > index2:
                index1, index2 = index2, index1
            parent1_genes_list = deepcopy(parent1.genes_list)
            parent2_genes_list:list = deepcopy(parent2.genes_list)

            child1_part = get_part(index1, index2, parent1_genes_list)
            for it in child1_part:
                parent2_genes_list.remove(it)
            child1_genes = parent2_genes_list[:index1] + child1_part + parent2_genes_list[index1:]

            return Chromosome(child1_genes, self.issue)

        offsprings = []
        for i in range(len(parents)):
            parents_list = np.random.choice(len(parents), 2, replace=False)
            offspring = ordered_crossover(parents[parents_list[0]], parents[parents_list[1]])
            offsprings.append(offspring)
            # offsprings.append(offspring2)
        return offsprings

    def selection(self, population, number):
        probs = []
        fitness_sum = sum(1 / chromosome.objective_function for chromosome in population.chromosomes)
        for i in range(len(population.chromosomes)):
            chromosome = population.chromosomes[i]
            prob = (1 / chromosome.objective_function) / fitness_sum
            probs.append(prob)
        chromo_index_list = np.random.choice(len(population.chromosomes), number, replace=False, p=probs)
        chromo_list = []
        for i in chromo_index_list:
            chromo_list.append(population.chromosomes[i])
        return chromo_list

    def solve(self):
        population = self.initial_population
        solution = self.get_best_chromosome(population)
        opt = solution.objective_function

        for i in tqdm.tqdm(range(50000)):
            parents = self.selection(population, 25)
            children = self.crossover(parents)
            mutated_chromo = []
            for j in range(len(children)):
                rand = random.uniform(0, 1)
                if rand <= 0.05:
                    mutated_chromo.append(self.mutation(children[j]))
                else:
                    mutated_chromo.append(children[j])
            population = parents + mutated_chromo
            population = Population(self.issue, population)
            cur_opt_chrom = self.get_best_chromosome(population)
            if cur_opt_chrom.objective_function < opt:
                opt = cur_opt_chrom.objective_function
                print(i, 'Best', opt)
                solution = cur_opt_chrom
        return solution

    def get_best_chromosome(self, population):
        # minimum = min(chromosome.objective_function for chromosome in population.chromosomes)
        best_chromosome = min(population.chromosomes, key=lambda x: x.objective_function)
        return best_chromosome


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

    def get_objective_function(self, issue):
        obj_func_value = 0
        for i in range(issue.dimension):
            for j in range(issue.dimension):
                obj_func_value += issue.distance_matrix[i][j] * issue.flow_matrix[self.genes_list[i]][
                    self.genes_list[j]]
        return obj_func_value
