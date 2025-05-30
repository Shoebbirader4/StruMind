from core.analysis.solver import FEMSolver
from core.geometry.analytical_model import AnalyticalModel, AnalyticalElement
import numpy as np

# Define nodes
node1 = (0, 0)
node2 = (5, 0)
node3 = (5, 3)

# Define elements (beams)
class SimpleBeam:
    def __init__(self, nodes):
        self.nodes = nodes
    def stiffness_matrix(self):
        # Dummy 6x6 matrix for demonstration
        return np.eye(6)

# Build model
model = AnalyticalModel()
model.nodes = [node1, node2, node3]
model.elements = [SimpleBeam([node1, node2]), SimpleBeam([node2, node3])]

# Define a static load at node 2, DOF 1 (vertical)
class DummyLoad:
    def __init__(self):
        self.location = 1  # node2
        self.direction = 1  # vertical
        self.magnitude = -10.0
loads = [DummyLoad()]

# Run analysis
solver = FEMSolver(model)
K = solver.assemble_global_stiffness()
print("Global stiffness matrix:\n", K)
displacements = solver.solve_static(loads)
print("Nodal displacements:", displacements) 