import re
import numpy as np


class Issue:
    def __init__(self, file_name):
        file = open('./test_instances/' + file_name)
        self.data = file.read().split('\n')
        self.dimension = int(re.findall('[0-9]+', self.data[0])[0])
        self.distance_data = self.data[1:self.dimension + 1]
        self.flow_data = self.data[self.dimension + 2:len(self.data)]

        self.distance_matrix = get_matrix(self.distance_data)
        self.flow_matrix = get_matrix(self.flow_data)


def get_matrix(data):
    matrix = []
    for i in range(0, len(data)):
        matrix_row = [int(element) for element in re.findall('[0-9]+', data[i])]
        matrix.append(matrix_row)
    return np.array(matrix)
