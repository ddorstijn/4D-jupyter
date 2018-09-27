from math import sqrt, tan
import numpy as np

from vector import VectorN

def main():
    print(create_simplex(4))

def create_simplex(dimensions):
    """ Generate vectors for a simplex in the nth dimension  """
    vertices = []
    # Return point if 0th dimension
    if dimensions == 0:
        vertices.append(VectorN([0]))
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
        # Solve the first missing coordinate
        vertices[i][i-1] = solve_dot_product(vertices, i, dimensions)
        # Solve the second missing coordinate
        vertices[i][i] = solve_magnitude(vertices, i)

    if dimensions > 3:
        project3d(vertices, dimensions)

    return vertices

def solve_dot_product(vertices, index, dimensions):
    """ Solve a1*b1 + a2*b2 + ... an+bn = -1/n """
    dot_product = sum([x**2 for x in vertices[index-1][:index-1]])
    dot_product = (-1.0/dimensions - dot_product) / vertices[index-1][index-1]
    
    return dot_product

def solve_magnitude(vertices, index):
    """ returns the length of the vector """
    return sqrt(1.0 - sum([x**2 for x in vertices[index]]))

def project3d(vertices, dimensions):
    """ project to lower dimension """
    view_matrix = get_projection_matrix(dimensions)

    for i, item in enumerate(vertices):
        vertices[i].homogeneous()
        np.matmul(vertices[i].data, view_matrix)
        vertices[i].normalize_reduce()
    
    if dimensions - 1 != 3:
        project3d(vertices, dimensions-1)

def get_lookat_matrix(dimensions, from_point, to_point):
    """ Return the matrix based on 2 vectors. 
        from_point(cam pos) and to_pont(where you look) """
    # Create a homogeneous matrix from the dimension
    matrix = np.identity(dimensions + 1)
    orthogonal_vectors = []

    for i in range(dimensions - 2):
        orthogonal_vectors.append(VectorN([0.0]*dimensions))
        for j in range(dimensions):
            if (i+1) == j:
                orthogonal_vectors[i][j] = 1.0

    to_point.subtract(from_point)
    columns = np.identity(dimensions)
    to_point.normalize()
    columns[dimensions-1] = to_point.data

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
                matrix[i][j] = columns[j][i]

    return matrix

def get_perspective_matrix(dimensions):
    """ Return the matrix that projects 
        the dimension onto a lower dimension """
    matrix = np.identity(dimensions + 1)
    fov = 1.0 / tan(90.0 / 2.0)

    # Fill the diagonal line with fov apart from the last 2
    for i in range(dimensions + 1):
        for j in range(dimensions+1):
            if i == j and i < dimensions - 1:
                matrix[i][j] = fov

    return matrix

def get_projection_matrix(dimensions):
    """ returns matrix by multiplying the 
        lookat with the perspective_matrix"""
    from_pos = VectorN([4.0, 4.0, 4, 1.0])
    to_pos = VectorN([0.0, 0.0, 0.0, 0.0])

    lookat_matrix = get_lookat_matrix(dimensions, from_pos, to_pos)
    perspective_matrix = get_perspective_matrix(dimensions)

    return np.matmul(lookat_matrix, perspective_matrix)

if __name__ == "__main__":
    main()

