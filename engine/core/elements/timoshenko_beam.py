import numpy as np

class TimoshenkoBeamElement:
    def __init__(self, start_node, end_node, section, material, shear_coeff=5/6, physical_geometry=None, fabrication_details=None):
        self.start_node = np.array(start_node)
        self.end_node = np.array(end_node)
        self.section = section
        self.material = material
        self.shear_coeff = shear_coeff
        self.physical_geometry = physical_geometry
        self.fabrication_details = fabrication_details or {}

    def length(self):
        return np.linalg.norm(self.end_node - self.start_node)

    def stiffness_matrix(self):
        # TODO: Implement Timoshenko beam element stiffness matrix
        # (includes shear deformation)
        L = self.length()
        E = self.material['E']
        I = self.section['I']
        A = self.section['A']
        kappa = self.shear_coeff
        G = self.material.get('G', E/(2*(1+self.material.get('nu',0.3))))
        # For demonstration, return a scaled identity
        return E * np.eye(6) 