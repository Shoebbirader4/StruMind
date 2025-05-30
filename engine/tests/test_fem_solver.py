import numpy as np
from core.analysis.solver import FEMSolver
from core.geometry.analytical_model import AnalyticalModel, AnalyticalElement

class DummyElement:
    def __init__(self, nodes):
        self.nodes = nodes
    def stiffness_matrix(self):
        # Simple 6x6 identity for 2 nodes (3 DOF each)
        return np.eye(6)

def test_fem_solver_static():
    model = AnalyticalModel()
    node1 = (0, 0)
    node2 = (5, 0)
    model.nodes = [node1, node2]
    elem = DummyElement([node1, node2])
    model.elements = [elem]
    solver = FEMSolver(model)
    K = solver.assemble_global_stiffness()
    assert K.shape == (6, 6)
    # Test static solve with a unit load at DOF 0
    class DummyLoad:
        def __init__(self):
            self.location = 0
            self.direction = 0
            self.magnitude = 1.0
    loads = [DummyLoad()]
    displacements = solver.solve_static(loads)
    assert displacements.shape == (6,) 