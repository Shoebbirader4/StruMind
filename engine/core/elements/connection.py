class Connection:
    def __init__(self, connection_type, connected_elements, geometry, fabrication_details=None):
        self.connection_type = connection_type  # e.g., 'bolted', 'welded', 'pinned', etc.
        self.connected_elements = connected_elements  # List of element references
        self.geometry = geometry
        self.fabrication_details = fabrication_details or {} 