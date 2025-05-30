import numpy as np
import cppsolver

# Example: 2x2 system
K = np.eye(2)
F = np.array([1.0, 2.0])
x = np.zeros(2)
cppsolver.solve_fem(K, F, x)
print('Solution from C++ FEM solver:', x) 