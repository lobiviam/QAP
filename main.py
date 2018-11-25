import issue
import ga

if __name__ == '__main__':
    file_name = 'tai20a'
    _issue = issue.Issue(file_name)
    solution = ga.GeneticAlg(_issue)