import math
import numpy as np

def main():
    print(create_simplex(3))

def create_simplex(dimensions):
    vertices = 
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
            vertices[i][j] = vertices[i-1][j];

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
        project3d(vertices)

    return vertices

def project3d(vertices):
    get_projection_matrix()

    for i, item in enumerate(vertices):
        vertices[i].homogenous()
        vertices[i].multiply_matrix(mat_view)
        vertices[i].normailize_reduce()

def get_projection_matrix():
    
    return pass

if __name__ == "__main__":
    main()

