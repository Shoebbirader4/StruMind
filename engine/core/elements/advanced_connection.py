class AdvancedConnection:
    def __init__(self, connection_type, connected_elements, geometry, parameters=None, fabrication_details=None):
        self.connection_type = connection_type
        self.connected_elements = connected_elements
        self.geometry = geometry
        self.parameters = parameters or {}
        self.fabrication_details = fabrication_details or {} 