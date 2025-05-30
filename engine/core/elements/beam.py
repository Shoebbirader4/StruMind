import numpy as np

class BeamElement:
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
        # 2D Euler-Bernoulli beam element (local coordinates)
        L = self.length()
        E = self.material['E']
        I = self.section['I']
        A = self.section['A']
        k = np.zeros((6, 6))
        # Axial terms
        k[0, 0] = k[3, 3] = E * A / L
        k[0, 3] = k[3, 0] = -E * A / L
        # Flexural terms
        k[1, 1] = k[4, 4] = 12 * E * I / L**3
        k[1, 4] = k[4, 1] = -12 * E * I / L**3
        k[1, 2] = k[2, 1] = 6 * E * I / L**2
        k[1, 5] = k[5, 1] = 6 * E * I / L**2
        k[2, 2] = k[5, 5] = 4 * E * I / L
        k[2, 4] = k[4, 2] = -6 * E * I / L**2
        k[2, 5] = k[5, 2] = 2 * E * I / L
        k[4, 5] = k[5, 4] = -6 * E * I / L**2
        # Symmetry
        k[3, 4] = k[4, 3] = 12 * E * I / L**3 * -1
        k[3, 5] = k[5, 3] = 6 * E * I / L**2 * -1
        # Transform to global coordinates
        c = (self.end_node[0] - self.start_node[0]) / L
        s = (self.end_node[1] - self.start_node[1]) / L
        T = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        k_global = T.T @ k @ T
        return k_global 

    def geometric_stiffness_matrix(self, N):
        """
        Geometric stiffness matrix for 2D beam element (global coordinates).
        N: axial force (positive = tension, negative = compression)
        Returns 6x6 matrix.
        """
        L = self.length()
        kG = np.zeros((6, 6))
        # Local geometric stiffness matrix for 2D beam (see structural analysis texts)
        coeff = N / (30 * L)
        kG_local = coeff * np.array([
            [36, 3*L, -36, 3*L, 0, 0],
            [3*L, 4*L**2, -3*L, -L**2, 0, 0],
            [-36, -3*L, 36, -3*L, 0, 0],
            [3*L, -L**2, -3*L, 4*L**2, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        # Transform to global coordinates
        c = (self.end_node[0] - self.start_node[0]) / L
        s = (self.end_node[1] - self.start_node[1]) / L
        T = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        kG_global = T.T @ kG_local @ T
        return kG_global 

    def mass_matrix(self, lumped=True):
        L = self.length()
        rho = self.material.get('rho', 7850)  # default steel
        A = self.section['A']
        m_total = rho * A * L
        if lumped:
            # Lumped mass: half to each node, only translational DOF
            m = np.zeros((6, 6))
            m[0, 0] = m[3, 3] = m_total / 2
            m[1, 1] = m[4, 4] = m_total / 2
            # Rotational DOF mass can be added if needed
            return m
        else:
            # Consistent mass matrix for 2D beam (translational only)
            m = (m_total / 420) * np.array([
                [140, 0, 0, 70, 0, 0],
                [0, 156, 22*L, 0, 54, -13*L],
                [0, 22*L, 4*L**2, 0, 13*L, -3*L**2],
                [70, 0, 0, 140, 0, 0],
                [0, 54, 13*L, 0, 156, -22*L],
                [0, -13*L, -3*L**2, 0, -22*L, 4*L**2]
            ])
            return m 