class PlateElement:
    def __init__(self, nodes, section, material, physical_geometry=None, fabrication_details=None):
        self.nodes = nodes  # List of nodes (typically 3 or 4 for shell/plate)
        self.section = section
        self.material = material
        self.physical_geometry = physical_geometry
        self.fabrication_details = fabrication_details or {} 