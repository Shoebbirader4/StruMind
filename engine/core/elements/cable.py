import numpy as np

class CableElement:
    def __init__(self, start_node, end_node, section, material, physical_geometry=None, fabrication_details=None):
        self.start_node = np.array(start_node)
        self.end_node = np.array(end_node)
        self.section = section
        self.material = material
        self.physical_geometry = physical_geometry
        self.fabrication_details = fabrication_details or {}

    def length(self):
        return np.linalg.norm(self.end_node - self.start_node)

    def stiffness_matrix(self):
        # Stub: 6x6 zero matrix (2 nodes x 3 dof)
        return np.zeros((6, 6))

    def geometric_stiffness_matrix(self, N):
        # Stub: 6x6 zero matrix
        return np.zeros((6, 6))

    def stiffness_matrix_global(self):
        # 2-node truss/cable element (global coordinates)
        L = self.length()
        E = self.material['E']
        A = self.section['A']
        c = (self.end_node[0] - self.start_node[0]) / L
        s = (self.end_node[1] - self.start_node[1]) / L
        k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
        T = np.array([[c, s, 0, 0], [0, 0, c, s]])
        # Expand to 4x4 global matrix
        k_global = np.zeros((4, 4))
        k_global[0:2, 0:2] = k_local
        k_global[2:4, 2:4] = k_local
        # Transform to global coordinates
        # For 2D truss, the transformation is simpler:
        T2 = np.array([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c],
        ])
        # For now, return the expanded local matrix (for demonstration)
        return k_global 