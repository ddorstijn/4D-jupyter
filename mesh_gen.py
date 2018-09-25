import math
import numpy as np

def main():
    print(create_simplex(4))

def create_simplex(dimensions):
    vertices = []
    # Return point if 0th dimension
    if dimensions == 0:
        vertices.append(VectorN(0))
        return vertices

    # For the first vertex we choose a point 1, 0, 0, ...
    vertices.append(VectorN([1.0] + [0.0]*(dimensions-1)))
    # Every other vertex can be generated in 2 steps
    for i in range(1, dimensions+1):
        vertices.append(VectorN([0]*dimensions))

        for j in range(i-1):
            vertices[i][j] = vertices[i-1][j]

        # The last vertex is special because it is just the negatve
        # of the vertex before
        if i == dimensions:
            vertices[i][i-1] = -vertices[i-1][i-1]
            break

        total_dot = sum([x**2 for x in vertices[i-1][:i-1]])
        total_dot = (-1.0/dimensions - total_dot) / vertices[i-1][i-1]
        vertices[i][i-1] = total_dot

        vertices[i][i] = math.sqrt(1.0 - sum([x**2 for x in vertices[i]]))
    
    if dimensions > 3:
        project3d(vertices, dimensions)

    return vertices

def project3d(vertices, dimensions):
    view_matrix = get_projection_matrix(dimensions)

    for i, item in enumerate(vertices):
        vertices[i].homogeneous()
        np.matmul(vertices[i].data, view_matrix)
        vertices[i].normalize_reduce()

def get_lookat_matrix(dimensions, from_point, to_point):
        matrix = np.identity(dimensions + 1)
        orthogonal_vectors = []

        for i in range(dimensions - 2):
            orthogonal_vectors.append(VectorN([0]*dimensions))
            for j in range(dimensions):
                orthogonal_vectors[i][j] = 1.0 if (i + 1.0) == j else 0.0
        
        to_point.subtract(from_point)
        columns = np.identity(dimensions)
        columns[dimensions-1] = to_point.normalize()

        for i in range(dimensions - 1):
            cross_vectors = []

            for j in range(dimensions -1):
                cross_vectors.append(VectorN([0]*dimensions))

            j = i - (dimensions - 2)
            for c in range(dimensions - 1):
                if j < 0:
                    cross_vectors[c] = orthogonal_vectors[(j + (dimensions - 2))]
                elif j == 0:
                    cross_vectors[c] = columns[(dimensions - 1)]
                else:
                    cross_vectors[c] = columns[(j - 1)]
                
                j += 1

            columns[i] = VectorN.get_normal(cross_vectors, dimensions).data

            if i != (dimensions - 2):
                np.linalg.norm(columns[i])

        for i in range(dimensions + 1):
            for j in range(dimensions + 1):
                if i < dimensions and j < dimensions:
                    matrix[i, j] = columns[j, i]
                elif i == j:
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0

        return matrix

def get_perspective_matrix(dimensions):
    matrix = np.identity(dimensions + 1)
    fov = 1.0 / math.tan(90 / 2.0)

    for i in range(dimensions + 1):
        for j in range(dimensions+1):
            if i == j:
                matrix[i][j] = 1.0 if i >= (dimensions - 1) else fov
            else:
                matrix[i][j] = 0.0

    return matrix

def get_projection_matrix(dimensions):
    from_pos = VectorN([4.0, 4.0, 4, 1.0])
    to_pos = VectorN([0.0, 0.0, 0.0, 0.0])

    lookat_matrix = get_lookat_matrix(dimensions, from_pos, to_pos)
    perspective_matrix = get_perspective_matrix(dimensions)

    return np.matmul(lookat_matrix, perspective_matrix)

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
        return math.sqrt(sum(el**2 for el in self.data))

    def normalize(self):
        length = self.magnitude()
        self.data = [el / length for el in self.data]

    def normalize_reduce(self):
        reducer = self.data[-1]
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
            base_vectors.append(VectorN([0]*dimensions))

        for i in range(dimensions - 1):
            for j in range(dimensions):
                # Fill the matrix with the iput vectors
                # in a column major style
                matrix[i][j] = input_vectors[i][j]
                # Fill the base vector 
                base_vectors[i][j] = 1.0 if (i == j) else 0.0
        
        for j in range(dimensions):
            i = dimensions - 1
            base_vectors[i][j] = 1.0 if (i == j) else 0.0
        
        normal_vector = VectorN([0]*dimensions)
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

if __name__ == "__main__":
    main()

