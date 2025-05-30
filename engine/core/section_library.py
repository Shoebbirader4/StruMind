import numpy as np

SECTION_LIBRARY = {
    'IPE200': {'A': 26.2e-4, 'I': 8570e-8, 'Z': 857e-6, 'type': 'steel'},
    'HEA200': {'A': 52.7e-4, 'I': 18600e-8, 'Z': 1860e-6, 'type': 'steel'},
    'Rect300x500': {'A': 0.15, 'I': 3.125e-3, 'Z': 1.25e-2, 'type': 'concrete'},
    # Add more sections as needed
}

class CustomSection:
    def __init__(self, name, params=None, polygon=None, section_type='custom'):
        self.name = name
        self.params = params or {}
        self.polygon = polygon  # list of (x, y) tuples if arbitrary shape
        self.type = section_type
        if polygon:
            self.A, self.I = self._compute_polygon_properties(polygon)
            self.Z = self.I / (max(abs(y) for x, y in polygon) or 1)
        else:
            # For common shapes, compute A, I, Z from params
            shape = params.get('shape', '')
            if shape == 'rect':
                b = params['b']
                h = params['h']
                self.A = b * h
                self.I = b * h**3 / 12
                self.Z = self.I / (h/2)
            elif shape == 'circle':
                r = params['r']
                self.A = np.pi * r**2
                self.I = np.pi/4 * r**4
                self.Z = self.I / r
            else:
                self.A = params.get('A', 0.01)
                self.I = params.get('I', 1e-6)
                self.Z = params.get('Z', 1e-4)
    def _compute_polygon_properties(self, polygon):
        # Shoelace formula for area, and centroidal I (approximate)
        x = np.array([pt[0] for pt in polygon])
        y = np.array([pt[1] for pt in polygon])
        n = len(polygon)
        A = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # Approximate I about centroid (not exact for all shapes)
        I = (1/12) * np.sum((x*np.roll(y,1) - y*np.roll(x,1)) * (x**2 + x*np.roll(x,1) + np.roll(x,1)**2))
        return abs(A), abs(I)
    def as_dict(self):
        return {'A': self.A, 'I': self.I, 'Z': self.Z, 'type': self.type, 'params': self.params, 'polygon': self.polygon}

def add_custom_section(name, params=None, polygon=None, section_type='custom'):
    cs = CustomSection(name, params, polygon, section_type)
    SECTION_LIBRARY[name] = cs.as_dict()
    return cs

def get_section(name):
    return SECTION_LIBRARY.get(name) 