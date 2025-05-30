import numpy as np
from core.elements.cable import CableElement
from core.analysis.solver import FEMSolver
from core.geometry.analytical_model import AnalyticalModel

# Define nodes
node1 = (0, 0)
node2 = (5, 0)
node3 = (2.5, 4)

# Define section and material
section = {'A': 0.01}
material = {'E': 210e9}

# Define elements (truss members)
elem1 = CableElement(node1, node2, section, material)
elem2 = CableElement(node2, node3, section, material)
elem3 = CableElement(node3, node1, section, material)

# Build model
model = AnalyticalModel()
model.nodes = [node1, node2, node3]
model.elements = [elem1, elem2, elem3]

# Define a static load at node 3, vertical direction
def make_load(node_idx, dof, magnitude):
    class DummyLoad:
        def __init__(self):
            self.location = node_idx
            self.direction = dof
            self.magnitude = magnitude
    return DummyLoad()

loads = [make_load(2, 1, -10000.0)]  # Node 3, vertical, -10kN

# Run analysis
solver = FEMSolver(model)
K = solver.assemble_global_stiffness()
print("Global stiffness matrix:\n", K)
displacements = solver.solve_static(loads)
print("Nodal displacements:", displacements) 