import numpy as np
from core.elements.beam import BeamElement
from core.elements.shell import ShellElement
from core.elements.solid import SolidElement
from core.analysis.solver import FEMSolver
from core.geometry.analytical_model import AnalyticalModel

# Nodes
n1 = (0, 0)
n2 = (5, 0)
n3 = (5, 5)
n4 = (0, 5)
n5 = (0, 0, 0)
n6 = (1, 0, 0)
n7 = (1, 1, 0)
n8 = (0, 1, 0)
n9 = (0, 0, 1)
n10 = (1, 0, 1)
n11 = (1, 1, 1)
n12 = (0, 1, 1)

# Section and material
section_beam = {'A': 0.01, 'I': 1e-6}
section_shell = {'A': 1.0, 'I': 1.0}
section_solid = {'A': 1.0, 'I': 1.0}
material = {'E': 210e9, 'nu': 0.3}

# Elements
beam = BeamElement(n1, n2, section_beam, material)
shell = ShellElement([n1, n2, n3, n4], section_shell, material, thickness=0.2)
solid = SolidElement([n5, n6, n7, n8, n9, n10, n11, n12], section_solid, material)

# Build model
model = AnalyticalModel()
model.nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12]
model.elements = [beam, shell, solid]

# Dummy load at node 2, vertical
def make_load(node_idx, dof, magnitude):
    class DummyLoad:
        def __init__(self):
            self.location = node_idx
            self.direction = dof
            self.magnitude = magnitude
    return DummyLoad()

loads = [make_load(1, 1, -5000.0)]

# Run analysis
solver = FEMSolver(model)
K = solver.assemble_global_stiffness()
print("Global stiffness matrix:\n", K)
displacements = solver.solve_static(loads)
print("Nodal displacements:", displacements) 