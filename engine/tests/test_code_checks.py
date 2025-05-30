from core.design.code_check import DesignCodeCheck

class DummyElement:
    def __init__(self, material, section, length=1.0):
        self.material = material
        self.section = section
        self._length = length
    def length(self):
        return self._length

def test_beam_code_checks():
    section = {'A': 1000, 'I': 8000, 'Z': 1000}
    material = {'E': 210e9, 'Fy': 250, 'fc': 30}
    forces = {'M': 200000, 'V': 10000, 'N': 50000}
    for code in ['AISC', 'Eurocode', 'ACI', 'IS']:
        check = DesignCodeCheck(code)
        result = check.check_beam(DummyElement(material, section), forces)
        assert isinstance(result, dict)
        assert 'bending' in result and 'shear' in result and 'axial' in result

def test_column_code_checks():
    section = {'A': 1000, 'I': 8000}
    material = {'E': 210e9, 'Fy': 250, 'fc': 30}
    forces = {'N': 50000, 'M': 200000}
    for code in ['AISC', 'Eurocode', 'ACI', 'IS']:
        check = DesignCodeCheck(code)
        result = check.check_column(DummyElement(material, section), forces)
        assert isinstance(result, dict)
        assert 'axial' in result and 'buckling' in result 