class PhysicalElement:
    def __init__(self, geometry, material, fabrication_details=None):
        self.geometry = geometry
        self.material = material
        self.fabrication_details = fabrication_details or {}

class PhysicalModel:
    def __init__(self):
        self.elements = []

    def add_element(self, element: PhysicalElement):
        self.elements.append(element) 