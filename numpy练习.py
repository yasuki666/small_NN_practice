import numpy as np
vector = np.array([1,2,3,4])
matrix = np.array([[1,2,3],
                   [1,2,3],
                   [1,2,3]])
print(vector)
print(matrix)

matrix2 = np.eye(3)
print(matrix2)

print(matrix-matrix2)
print(matrix+matrix2)
print(matrix*matrix2)

print(matrix.shape[1]==matrix2.shape[0])
print(matrix.dot(matrix2))

print((matrix.T))

A = np.array([[0,1],[2,3]])
invA = np.linalg.inv(A)
print(invA)
