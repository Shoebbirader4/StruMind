import numpy as np
from core.elements.shell import ShellElement
from core.elements.solid import SolidElement

def test_shell_element_stiffness():
    nodes = [(0,0), (2,0), (2,1), (0,1)]
    section = {'A': 1.0, 'I': 1.0}
    material = {'E': 210e9, 'nu': 0.3}
    shell = ShellElement(nodes, section, material, thickness=0.2)
    k = shell.stiffness_matrix()
    assert k.shape == (12, 12)
    assert np.allclose(k, k.T)  # Should be symmetric

def test_solid_element_stiffness():
    nodes = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    ]
    section = {'A': 1.0, 'I': 1.0}
    material = {'E': 30e9, 'nu': 0.2}
    solid = SolidElement(nodes, section, material)
    k = solid.stiffness_matrix()
    assert k.shape == (24, 24)
    assert np.allclose(k, k.T)  # Should be symmetric 