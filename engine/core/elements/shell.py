import numpy as np

class ShellElement:
    def __init__(self, nodes, section, material, thickness=1.0, physical_geometry=None, fabrication_details=None):
        self.nodes = [np.array(n) for n in nodes]  # 4 nodes (x, y)
        self.section = section
        self.material = material
        self.thickness = thickness
        self.physical_geometry = physical_geometry
        self.fabrication_details = fabrication_details or {}

    def stiffness_matrix(self):
        # Stub: 12x12 zero matrix (4 nodes x 3 dof)
        return np.zeros((12, 12))

    def geometric_stiffness_matrix(self, N):
        # Stub: 12x12 zero matrix
        return np.zeros((12, 12))

    def stiffness_matrix_real(self):
        # Simple 4-node rectangular plate (Kirchhoff/MITC4, demonstration only)
        # Assumes nodes are ordered: (0,0), (a,0), (a,b), (0,b)
        E = self.material['E']
        nu = self.material.get('nu', 0.3)
        t = self.thickness
        a = np.linalg.norm(self.nodes[1] - self.nodes[0])
        b = np.linalg.norm(self.nodes[3] - self.nodes[0])
        D = E * t**3 / (12 * (1 - nu**2))
        # Very simplified stiffness matrix for demonstration (real MITC4 is much more complex)
        k = D * np.eye(12) / (a * b)
        return k 

    def mass_matrix(self, lumped=True):
        rho = self.material.get('rho', 2500)  # default concrete
        t = self.thickness
        a = np.linalg.norm(self.nodes[1] - self.nodes[0])
        b = np.linalg.norm(self.nodes[3] - self.nodes[0])
        area = a * b
        m_total = rho * area * t
        if lumped:
            m = np.zeros((12, 12))
            for i in range(4):
                m[i*3, i*3] = m_total / 4
                m[i*3+1, i*3+1] = m_total / 4
                m[i*3+2, i*3+2] = m_total / 4
            return m
        else:
            # Consistent mass matrix for a rectangular shell (simplified)
            m = (m_total / 36) * np.eye(12)
            return m 