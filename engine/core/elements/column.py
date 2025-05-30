import numpy as np
from engine.core.elements.beam import BeamElement

class ColumnElement(BeamElement):
    """
    Column element (inherits from BeamElement for now).
    """
    def __init__(self, base_node, top_node, section, material, physical_geometry=None, fabrication_details=None):
        super().__init__(base_node, top_node, section, material, physical_geometry, fabrication_details)
        self.base_node = base_node
        self.top_node = top_node
        self.section = section
        self.material = material
        self.physical_geometry = physical_geometry
        self.fabrication_details = fabrication_details or {} 