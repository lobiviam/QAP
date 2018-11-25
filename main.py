import issue
import ga


def print_solution_if_file(file_name, solution):
    with open(file_name + '.sol', 'w') as out_file:
        for gene in solution.genes_list:
            out_file.write(str(gene+1) + ' ')


if __name__ == '__main__':
    file_name = 'tai80a'
    _issue = issue.Issue(file_name)
    _solution = ga.GeneticAlg(_issue).solution
    print(_solution.genes_list)
    print_solution_if_file(file_name, _solution)
