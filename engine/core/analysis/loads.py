class Load:
    def __init__(self, load_type, magnitude, location, direction=None, span=None, area=None, temperature=None):
        self.load_type = load_type  # 'point', 'distributed', 'area', 'temperature'
        self.magnitude = magnitude
        self.location = location  # node index or element index
        self.direction = direction  # dof or vector
        self.span = span  # (start, end) for distributed
        self.area = area  # (element, value) for area loads
        self.temperature = temperature  # for temperature loads

class LoadCase:
    def __init__(self, name):
        self.name = name
        self.loads = []

    def add_load(self, load: Load):
        self.loads.append(load)

class LoadCombination:
    def __init__(self, name):
        self.name = name
        self.cases = {}  # {LoadCase: factor}

    def add_case(self, load_case, factor):
        self.cases[load_case] = factor

    @staticmethod
    def generate_combinations(load_cases, code='ASCE'):  # stub for code-based combos
        # Example: for ASCE, generate 1.2*DL + 1.6*LL, etc.
        combos = []
        if code == 'ASCE':
            for lc1 in load_cases:
                for lc2 in load_cases:
                    if lc1.name == 'DL' and lc2.name == 'LL':
                        combo = LoadCombination('1.2DL+1.6LL')
                        combo.add_case(lc1, 1.2)
                        combo.add_case(lc2, 1.6)
                        combos.append(combo)
        # Add more codes/combinations as needed
        return combos 