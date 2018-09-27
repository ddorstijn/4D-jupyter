import numpy as np
from math import sqrt

class VectorN:
    data = []

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def add(self, other):
        self.data = [a + b for a,b in zip(self.data, other.data)]

    def subtract(self, other):
        self.data = [a - b for a,b in zip(self.data, other.data)]

    def multiply(self, other):
        if other is VectorN:
            self.data = [a * b for a,b in zip(self.data, other.data)]

    def divide(self, other):
        self.data = [a / b for a,b in zip(self.data, other.data)]

    def scale(self, scalar):
        self.data = [el * scalar for el in self.data]

    def magnitude(self):
        return sqrt(sum(el**2 for el in self.data))

    def normalize(self):
        length = self.magnitude()
        self.data = [el / length for el in self.data]

    def normalize_reduce(self):
        reducer = self.data[-1]
        if reducer != 0:
            self.data = [el / reducer for el in self.data[:-1]]
        self.data.pop()

    def homogeneous(self):
        self.data.append(1.0)


    @staticmethod
    def get_normal(input_vectors, dimensions):
        # The matrix we apply the vectors to
        matrix = np.identity(dimensions)
        # Base vectors are all unit vectors for the dimension over all it's axis
        # Example 3D: 1 0 0, 0 1 0, 0 0 1 
        base_vectors = []

        for i in range(dimensions):
            base_vectors.append(VectorN([0.0]*dimensions))

        for i in range(dimensions - 1):
            for j in range(dimensions):
                # Fill the matrix with the iput vectors
                # in a column major style
                matrix[i][j] = input_vectors[i][j]
                # Fill the base vector 
                if i == j:
                    base_vectors[i][j] = 1.0

        base_vectors[-1][-1] = 1.0

        normal_vector = VectorN([0.0]*dimensions)
        for i in range(dimensions):
            s = np.identity(dimensions - 1)
            r = 0
            for j in range(dimensions - 1):
                c = 0
                for k in range(dimensions):
                    if k == i:
                        continue

                    s[r][c] = matrix[j][k]
                    c += 1
                    k += 1
                r += 1

            if (i % 2) == 0:
                base_vectors[i].scale(np.linalg.det(s))
                normal_vector.add(base_vectors[i])
            else:
                base_vectors[i].scale(np.linalg.det(s))
                normal_vector.subtract(base_vectors[i])

        return normal_vector

