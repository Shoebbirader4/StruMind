MATERIAL_LIBRARY = {
    'S235': {'E': 210e9, 'Fy': 235e6, 'type': 'steel'},
    'S355': {'E': 210e9, 'Fy': 355e6, 'type': 'steel'},
    'C30/37': {'E': 33e9, 'fc': 30e6, 'type': 'concrete'},
    'C40/50': {'E': 35e9, 'fc': 40e6, 'type': 'concrete'},
    # Add more materials as needed
}

def get_material(name):
    return MATERIAL_LIBRARY.get(name) 