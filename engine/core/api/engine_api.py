from engine.core.analysis.solver import FEMSolver
from engine.core.design.code_check import DesignCodeCheck
from engine.core.geometry.analytical_model import AnalyticalModel
from engine.core.section_library import SECTION_LIBRARY, get_section
from engine.core.material_library import MATERIAL_LIBRARY, get_material
from engine.core.analysis.loads import Load, LoadCase, LoadCombination
import ifcopenshell
import ifcopenshell.api
import numpy as np
import svgwrite
import csv
import os
try:
    import reportlab
    from reportlab.pdfgen import canvas
except ImportError:
    reportlab = None
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None

class EngineAPI:
    def __init__(self):
        self.model = None
        self.solver = None
        self.results = None
        self.code_check_results = None

    def load_model(self, model: AnalyticalModel):
        self.model = model
        self.solver = FEMSolver(model)
        self.results = None
        self.code_check_results = None

    def run_analysis(self, loads):
        if not self.solver:
            raise RuntimeError('No model loaded')
        K = self.solver.assemble_global_stiffness()
        displacements = self.solver.solve_static(loads)
        self.results = {'K': K, 'displacements': displacements}
        return self.results

    def run_code_checks(self, code_name, forces_by_element):
        if not self.model:
            raise RuntimeError('No model loaded')
        checker = DesignCodeCheck(code_name)
        self.code_check_results = {}
        for i, elem in enumerate(self.model.elements):
            forces = forces_by_element.get(i, {})
            if hasattr(checker, 'check_beam'):
                self.code_check_results[i] = checker.check_beam(elem, forces)
        return self.code_check_results

    def get_results(self):
        return self.results

    def get_code_check_results(self):
        return self.code_check_results

    def get_section_names(self):
        return list(SECTION_LIBRARY.keys())

    def get_material_names(self):
        return list(MATERIAL_LIBRARY.keys())

    def get_section(self, name):
        return get_section(name)

    def get_material(self, name):
        return get_material(name)

    def create_load_case(self, name, loads):
        lc = LoadCase(name)
        for load in loads:
            lc.add_load(load)
        return lc

    def create_load_combination(self, name, cases_and_factors):
        combo = LoadCombination(name)
        for lc, factor in cases_and_factors:
            combo.add_case(lc, factor)
        return combo

    def generate_code_combinations(self, load_cases, code='ASCE'):
        return LoadCombination.generate_combinations(load_cases, code)

    def get_beam_internal_force_diagrams(self):
        if not self.solver or not self.results:
            return None
        return self.solver.compute_beam_internal_forces(self.results['displacements'])

    def get_shell_displacement_contours(self):
        if not self.solver or not self.results:
            return None
        return self.solver.compute_shell_displacement_contour(self.results['displacements'])

    def run_nonlinear_analysis(self, options=None):
        # Accept options: steps, max_iter, tol
        if not self.solver:
            raise RuntimeError('No model loaded')
        if options is None:
            options = {'steps': 10, 'max_iter': 20, 'tol': 1e-4}
        # Call the solver's nonlinear method (stub for now)
        result = self.solver.solve_nonlinear(options)
        return result

    def run_dynamic_analysis(self, loads):
        # Stub for dynamic analysis
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_dynamic()

    def optimize_sections(self):
        # Stub for section optimization
        pass

    def create_beam(self, *args, **kwargs):
        # TODO: Create and add a beam element
        pass

    def create_column(self, *args, **kwargs):
        # TODO: Create and add a column element
        pass

    def run_design(self):
        # TODO: Run design checks
        pass

    def export_ifc(self, filename=None):
        if filename is None:
            return 'No IFC file specified.'
        try:
            ifc = ifcopenshell.file(schema='IFC4')
            project = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcProject', name='ExportedProject')
            site = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcSite', name='DefaultSite')
            building = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcBuilding', name='DefaultBuilding')
            storey = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcBuildingStorey', name='DefaultStorey')
            ifcopenshell.api.run('aggregate.assign_object', ifc, product=project, relating_object=site)
            ifcopenshell.api.run('aggregate.assign_object', ifc, product=site, relating_object=building)
            ifcopenshell.api.run('aggregate.assign_object', ifc, product=building, relating_object=storey)
            for elem in getattr(self.model, 'elements', []):
                if hasattr(elem, 'start_node') and hasattr(elem, 'end_node'):
                    beam = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcBeam', name=getattr(elem, 'name', 'Beam'))
                    # Section profile
                    prof = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcRectangleProfileDef', name='Profile', XDim=elem.section.get('b', 0.1), YDim=elem.section.get('h', 0.2))
                    # Material
                    mat = ifcopenshell.api.run('material.add_material', ifc, name='Steel')
                    ifcopenshell.api.run('spatial.assign_container', ifc, product=beam, relating_structure=storey)
                elif hasattr(elem, 'nodes') and len(elem.nodes) == 4:
                    plate = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcPlate', name=getattr(elem, 'name', 'Shell'))
                    mat = ifcopenshell.api.run('material.add_material', ifc, name='Concrete')
                    ifcopenshell.api.run('spatial.assign_container', ifc, product=plate, relating_structure=storey)
                elif hasattr(elem, 'nodes') and len(elem.nodes) == 8:
                    member = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcMember', name=getattr(elem, 'name', 'Solid'))
                    mat = ifcopenshell.api.run('material.add_material', ifc, name='Concrete')
                    ifcopenshell.api.run('spatial.assign_container', ifc, product=member, relating_structure=storey)
                elif hasattr(elem, 'start_node') and hasattr(elem, 'end_node') and hasattr(elem, 'section') and elem.section['A'] < 0.005:
                    cable = ifcopenshell.api.run('root.create_entity', ifc, ifc_class='IfcCableSegment', name=getattr(elem, 'name', 'Cable'))
                    mat = ifcopenshell.api.run('material.add_material', ifc, name='Steel')
                    ifcopenshell.api.run('spatial.assign_container', ifc, product=cable, relating_structure=storey)
            ifc.write(filename)
            return f'Exported {len(getattr(self.model, "elements", []))} elements to IFC.'
        except Exception as e:
            return f'IFC export error: {e}'

    def run_buckling_analysis(self):
        if not self.solver:
            raise RuntimeError('No model loaded')
        crit_lambda, crit_mode = self.solver.compute_critical_buckling_load()
        return {'critical_load_factor': crit_lambda, 'mode_shape': crit_mode}

    def run_modal_analysis(self, num_modes=5, lumped=True):
        """
        Run modal analysis and return natural frequencies and mode shapes.
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.compute_modal_analysis(num_modes=num_modes, lumped=lumped)

    def run_time_history(self, F_time, dt=0.01, t_total=1.0, beta=0.25, gamma=0.5, alpha=0.0, beta_rayleigh=0.01, lumped=True, C=None, M=None):
        """
        Run time history analysis using Newmark-beta method.
        F_time: function or array, F(t) for each time step (shape: [n_steps, n_dof])
        Returns: u_hist, v_hist, a_hist (displacement, velocity, acceleration time histories)
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_time_history(F_time, dt, t_total, beta, gamma, alpha, beta_rayleigh, lumped, C, M)

    def run_response_spectrum(self, spectrum, num_modes=5, combine='SRSS', lumped=True):
        """
        Run response spectrum analysis (SRSS modal combination).
        spectrum: function or array, giving spectral acceleration vs. period/frequency
        Returns: peak response at each DOF
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_response_spectrum(spectrum, num_modes, combine, lumped)

    def get_mass_matrix(self, lumped=True):
        """
        Return the assembled global mass matrix.
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.assemble_global_mass(lumped=lumped)

    def get_damping_matrix(self, alpha=0.0, beta=0.01, lumped=True):
        """
        Return the assembled global damping matrix (Rayleigh damping).
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.assemble_global_damping(alpha, beta, lumped=lumped)

    def import_ifc(self, filename=None):
        if filename is None:
            return 'No IFC file specified.'
        try:
            ifc = ifcopenshell.open(filename)
            model = AnalyticalModel()
            node_map = {}
            def add_node(p):
                if p not in node_map:
                    model.add_node(p)
                    node_map[p] = len(model.nodes) - 1
            def parse_profile(profile):
                # Parse IfcProfileDef for section properties
                if profile.is_a('IfcRectangleProfileDef'):
                    b = float(profile.XDim)
                    h = float(profile.YDim)
                    A = b * h
                    I = (b * h**3) / 12
                    return {'A': A, 'I': I, 'b': b, 'h': h}
                elif profile.is_a('IfcCircleProfileDef'):
                    r = float(profile.Radius)
                    A = np.pi * r**2
                    I = (np.pi/4) * r**4
                    return {'A': A, 'I': I, 'r': r}
                elif profile.is_a('IfcIShapeProfileDef'):
                    b = float(profile.OverallWidth)
                    h = float(profile.OverallDepth)
                    tw = float(profile.WebThickness)
                    tf = float(profile.FlangeThickness)
                    A = b*tf*2 + (h-2*tf)*tw
                    I = (b*h**3 - (b-tw)*(h-2*tf)**3) / 12
                    return {'A': A, 'I': I, 'b': b, 'h': h, 'tw': tw, 'tf': tf}
                elif profile.is_a('IfcTShapeProfileDef'):
                    b = float(profile.FlangeWidth)
                    h = float(profile.Depth)
                    tw = float(profile.WebThickness)
                    tf = float(profile.FlangeThickness)
                    A = b*tf + (h-tf)*tw
                    I = (b*tf**3/12) + (tw*(h-tf)**3/12)
                    return {'A': A, 'I': I, 'b': b, 'h': h, 'tw': tw, 'tf': tf}
                elif profile.is_a('IfcLShapeProfileDef'):
                    b = float(profile.Width)
                    h = float(profile.Depth)
                    tw = float(profile.Thickness)
                    A = b*tw + (h-tw)*tw
                    I = (b*tw**3/12) + (tw*(h-tw)**3/12)
                    return {'A': A, 'I': I, 'b': b, 'h': h, 'tw': tw}
                elif profile.is_a('IfcUShapeProfileDef'):
                    b = float(profile.FlangeWidth)
                    h = float(profile.Depth)
                    tw = float(profile.WebThickness)
                    tf = float(profile.FlangeThickness)
                    A = 2*b*tf + (h-2*tf)*tw
                    I = (b*h**3 - (b-tw)*(h-2*tf)**3) / 12
                    return {'A': A, 'I': I, 'b': b, 'h': h, 'tw': tw, 'tf': tf}
                elif profile.is_a('IfcPipeProfileDef'):
                    r = float(profile.Radius)
                    t = float(profile.WallThickness)
                    A = np.pi * (r**2 - (r-t)**2)
                    I = (np.pi/4) * (r**4 - (r-t)**4)
                    return {'A': A, 'I': I, 'r': r, 't': t}
                # Add more profiles as needed
                return {'A': 0.01, 'I': 1e-6}
            def parse_material(mat):
                # Parse IfcMaterial for E, Fy, density
                # Try to extract from IfcMaterialPropertySet
                E = 210e9
                Fy = 250
                rho = 7850
                if hasattr(mat, 'HasProperties'):
                    for prop in mat.HasProperties:
                        if hasattr(prop, 'Name') and prop.Name:
                            n = prop.Name.lower()
                            if 'modulus' in n or 'e' == n:
                                try: E = float(prop.NominalValue.wrappedValue)
                                except: pass
                            if 'fy' in n or 'yield' in n:
                                try: Fy = float(prop.NominalValue.wrappedValue)
                                except: pass
                            if 'density' in n or 'rho' in n:
                                try: rho = float(prop.NominalValue.wrappedValue)
                                except: pass
                return {'E': E, 'Fy': Fy, 'rho': rho}
            # Beams/Columns
            for ifc_elem in ifc.by_type('IfcBeam') + ifc.by_type('IfcColumn'):
                try:
                    rep = ifc_elem.Representation.Representations[0]
                    items = rep.Items
                    # Geometry: IfcExtrudedAreaSolid
                    if hasattr(items[0], 'SweptArea') and hasattr(items[0], 'Depth'):
                        profile = items[0].SweptArea
                        section = parse_profile(profile)
                        depth = float(items[0].Depth)
                        # Start point
                        pos = items[0].Position.Location
                        p1 = tuple(float(x) for x in pos.Coordinates)
                        # End point (extrude along Z by depth)
                        dir = items[0].ExtrudedDirection.DirectionRatios
                        p2 = tuple(p1[i] + depth * dir[i] for i in range(3))
                    elif hasattr(items[0], 'Points'):
                        pts = items[0].Points
                        p1 = tuple(float(x) for x in pts[0].Coordinates)
                        p2 = tuple(float(x) for x in pts[1].Coordinates)
                        section = {'A': 0.01, 'I': 1e-6}
                    else:
                        continue
                except Exception:
                    continue
                add_node(p1)
                add_node(p2)
                # Material
                material = {'E': 210e9, 'Fy': 250, 'rho': 7850}
                if hasattr(ifc_elem, 'HasAssociations') and ifc_elem.HasAssociations:
                    for assoc in ifc_elem.HasAssociations:
                        if assoc.is_a('IfcRelAssociatesMaterial'):
                            mats = assoc.RelatingMaterial
                            if hasattr(mats, 'Name'):
                                material = parse_material(mats)
                name = getattr(ifc_elem, 'Name', 'Beam/Column')
                from engine.core.elements.beam import BeamElement
                elem = BeamElement(p1, p2, section, material)
                elem.name = name
                model.add_element(elem)
            # Plates (Shells)
            for ifc_elem in ifc.by_type('IfcPlate'):
                try:
                    rep = ifc_elem.Representation.Representations[0]
                    items = rep.Items
                    if hasattr(items[0], 'Points'):
                        pts = items[0].Points
                        nodes = [tuple(float(x) for x in pt.Coordinates) for pt in pts]
                        if len(nodes) != 4:
                            continue
                    else:
                        continue
                except Exception:
                    continue
                for p in nodes:
                    add_node(p)
                section = {'A': 0.01, 'I': 1e-6}
                material = {'E': 210e9, 'Fy': 250, 'rho': 2500}
                name = getattr(ifc_elem, 'Name', 'Shell')
                from engine.core.elements.shell import ShellElement
                elem = ShellElement(nodes, section, material)
                elem.name = name
                model.add_element(elem)
            # Solids (IfcMember)
            for ifc_elem in ifc.by_type('IfcMember'):
                try:
                    rep = ifc_elem.Representation.Representations[0]
                    items = rep.Items
                    if hasattr(items[0], 'Points'):
                        pts = items[0].Points
                        nodes = [tuple(float(x) for x in pt.Coordinates) for pt in pts]
                        if len(nodes) != 8:
                            continue
                    else:
                        continue
                except Exception:
                    continue
                for p in nodes:
                    add_node(p)
                section = {'A': 0.01, 'I': 1e-6}
                material = {'E': 210e9, 'Fy': 250, 'rho': 2500}
                name = getattr(ifc_elem, 'Name', 'Solid')
                from engine.core.elements.solid import SolidElement
                elem = SolidElement(nodes, section, material)
                elem.name = name
                model.add_element(elem)
            # Cables (IfcTendon/IfcCableSegment)
            for ifc_elem in ifc.by_type('IfcTendon') + ifc.by_type('IfcCableSegment'):
                try:
                    rep = ifc_elem.Representation.Representations[0]
                    items = rep.Items
                    if hasattr(items[0], 'Points'):
                        pts = items[0].Points
                        p1 = tuple(float(x) for x in pts[0].Coordinates)
                        p2 = tuple(float(x) for x in pts[1].Coordinates)
                    else:
                        continue
                except Exception:
                    continue
                add_node(p1)
                add_node(p2)
                section = {'A': 0.001, 'I': 1e-8}
                material = {'E': 200e9, 'Fy': 1770e6, 'rho': 7850}
                name = getattr(ifc_elem, 'Name', 'Cable')
                from engine.core.elements.cable import CableElement
                elem = CableElement(p1, p2, section, material)
                elem.name = name
                model.add_element(elem)
            self.load_model(model)
            return f'Imported {len(model.elements)} elements from IFC.'
        except Exception as e:
            return f'IFC import error: {e}'

    def generate_drawings(self, filename=None, view='plan', overlay_forces=True, pdf=False):
        if filename is None:
            return 'No drawing file specified.'
        try:
            if pdf and reportlab:
                c = canvas.Canvas(filename)
                # Draw elements (plan or elevation)
                for i, elem in enumerate(getattr(self.model, 'elements', [])):
                    if hasattr(elem, 'nodes') and len(elem.nodes) >= 2:
                        n1 = elem.nodes[0]
                        n2 = elem.nodes[1]
                        if view == 'plan':
                            x1, y1 = n1[0]*100, n1[1]*100
                            x2, y2 = n2[0]*100, n2[1]*100
                        elif view == 'elevation':
                            x1, y1 = n1[0]*100, n1[2]*100 if len(n1) > 2 else 0
                            x2, y2 = n2[0]*100, n2[2]*100 if len(n2) > 2 else 0
                        else:
                            x1, y1 = n1[0]*100, n1[1]*100
                            x2, y2 = n2[0]*100, n2[1]*100
                        c.line(x1, y1, x2, y2)
                        c.drawString((x1+x2)/2, (y1+y2)/2, f'E{i+1}')
                c.save()
                return f'PDF drawing exported to {filename}'
            else:
                import svgwrite
                dwg = svgwrite.Drawing(filename, profile='tiny')
                for i, elem in enumerate(getattr(self.model, 'elements', [])):
                    if hasattr(elem, 'nodes') and len(elem.nodes) >= 2:
                        n1 = elem.nodes[0]
                        n2 = elem.nodes[1]
                        if view == 'plan':
                            x1, y1 = n1[0]*100, -n1[1]*100
                            x2, y2 = n2[0]*100, -n2[1]*100
                        elif view == 'elevation':
                            x1, y1 = n1[0]*100, -n1[2]*100 if len(n1) > 2 else 0
                            x2, y2 = n2[0]*100, -n2[2]*100 if len(n2) > 2 else 0
                        else:
                            x1, y1 = n1[0]*100, -n1[1]*100
                            x2, y2 = n2[0]*100, -n2[1]*100
                        dwg.add(dwg.line((x1, y1), (x2, y2), stroke=svgwrite.rgb(10, 10, 16, '%')))
                        mx = (x1+x2)/2
                        my = (y1+y2)/2
                        dwg.add(dwg.text(f'E{i+1}', insert=(mx, my), fill='red', font_size='10px'))
                # Overlay force diagrams if available
                if overlay_forces and self.results and 'displacements' in self.results:
                    diagrams = self.get_beam_internal_force_diagrams()
                    if diagrams:
                        for d in diagrams:
                            x = d['x']
                            M = d['M']
                            elem = d['element']
                            n1, n2 = elem.nodes
                            # Draw moment diagram (simple straight line for demo)
                            for j in range(len(x)-1):
                                x0 = n1[0]*100 + (n2[0]-n1[0])*100*x[j]/x[-1]
                                y0 = -n1[1]*100 - M[j]*0.01
                                x1 = n1[0]*100 + (n2[0]-n1[0])*100*x[j+1]/x[-1]
                                y1 = -n1[1]*100 - M[j+1]*0.01
                                dwg.add(dwg.line((x0, y0), (x1, y1), stroke='green'))
                dwg.save()
                return f'Drawing exported to {filename}'
        except Exception as e:
            return f'Drawing export error: {e}'

    def generate_report(self, filename=None, pdf=False, excel=False):
        if filename is None:
            return 'No report file specified.'
        try:
            if pdf and reportlab:
                c = canvas.Canvas(filename)
                y = 800
                c.drawString(50, y, 'Nodes:')
                y -= 20
                for i, n in enumerate(getattr(self.model, 'nodes', [])):
                    c.drawString(50, y, f'N{i+1}: {n}')
                    y -= 15
                y -= 10
                c.drawString(50, y, 'Elements:')
                y -= 20
                for i, elem in enumerate(getattr(self.model, 'elements', [])):
                    c.drawString(50, y, f'E{i+1}: {type(elem).__name__}, Nodes: {getattr(elem, "nodes", [])}')
                    y -= 15
                y -= 10
                if self.code_check_results:
                    c.drawString(50, y, 'Code Check Results:')
                    y -= 20
                    for k, v in self.code_check_results.items():
                        c.drawString(50, y, f'Element {k+1}: {v}')
                        y -= 15
                c.save()
                return f'PDF report exported to {filename}'
            elif excel and xlsxwriter:
                wb = xlsxwriter.Workbook(filename)
                ws = wb.add_worksheet('Nodes')
                ws.write_row(0, 0, ['Index', 'X', 'Y', 'Z'])
                for i, n in enumerate(getattr(self.model, 'nodes', [])):
                    ws.write_row(i+1, 0, [i+1] + list(n) + [0]*(3-len(n)))
                ws2 = wb.add_worksheet('Elements')
                ws2.write_row(0, 0, ['Index', 'Type', 'Node1', 'Node2', 'Node3', 'Node4', 'Section', 'Material'])
                for i, elem in enumerate(getattr(self.model, 'elements', [])):
                    typ = type(elem).__name__
                    nodes = getattr(elem, 'nodes', [])
                    node_ids = [self.model.nodes.index(tuple(n)) + 1 for n in nodes]
                    ws2.write_row(i+1, 0, [i+1, typ] + node_ids + ['']*(4-len(node_ids)) + [str(getattr(elem, 'section', {})), str(getattr(elem, 'material', {}))])
                if self.code_check_results:
                    ws3 = wb.add_worksheet('CodeChecks')
                    ws3.write_row(0, 0, ['Element', 'Result'])
                    for i, (k, v) in enumerate(self.code_check_results.items()):
                        ws3.write_row(i+1, 0, [k+1, str(v)])
                wb.close()
                return f'Excel report exported to {filename}'
            else:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Nodes'])
                    writer.writerow(['Index', 'X', 'Y', 'Z'])
                    for i, n in enumerate(getattr(self.model, 'nodes', [])):
                        row = [i+1] + list(n) + [0]*(3-len(n))
                        writer.writerow(row)
                    writer.writerow([])
                    writer.writerow(['Elements'])
                    writer.writerow(['Index', 'Type', 'Node1', 'Node2', 'Node3', 'Node4', 'Section', 'Material'])
                    for i, elem in enumerate(getattr(self.model, 'elements', [])):
                        typ = type(elem).__name__
                        nodes = getattr(elem, 'nodes', [])
                        node_ids = [self.model.nodes.index(tuple(n)) + 1 for n in nodes]
                        row = [i+1, typ] + node_ids + ['']*(4-len(node_ids)) + [getattr(elem, 'section', {}), getattr(elem, 'material', {})]
                        writer.writerow(row)
                    if self.code_check_results:
                        writer.writerow([])
                        writer.writerow(['Code Check Results'])
                        writer.writerow(['Element', 'Result'])
                        for k, v in self.code_check_results.items():
                            writer.writerow([k+1, v])
                    if self.results and 'displacements' in self.results:
                        writer.writerow([])
                        writer.writerow(['Displacements'])
                        writer.writerow(['DOF', 'Value'])
                        for i, val in enumerate(self.results['displacements']):
                            writer.writerow([i+1, val])
                return f'Report exported to {filename}'
        except Exception as e:
            return f'Report export error: {e}'

    def undo(self):
        # Stub for undo
        pass

    def redo(self):
        # Stub for redo
        pass

    def run_staged_construction(self, stages):
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_staged_construction(stages)

    def run_pushover_analysis(self, control_node, direction=1, steps=10, max_disp=0.1):
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_pushover(control_node, direction, steps, max_disp)

    def run_time_history_analysis(self, F_time, dt=0.01, t_total=1.0, beta=0.25, gamma=0.5, C=None, M=None):
        """
        Run time history analysis (Newmark-beta).
        F_time: function or array, F(t) for each time step (shape: [n_steps, n_dof])
        dt: time step
        t_total: total time
        beta, gamma: Newmark parameters
        C: damping matrix (optional)
        M: mass matrix (optional)
        Returns: u_hist, v_hist, a_hist
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_time_history(F_time, dt, t_total, beta, gamma, C, M)

    def run_response_spectrum_analysis(self, spectrum, num_modes=5, combine='SRSS'):
        """
        Run response spectrum analysis.
        spectrum: function or array, giving spectral acceleration vs. period/frequency
        num_modes: number of modes to use
        combine: 'SRSS' or 'CQC'
        Returns: peak response at each DOF
        """
        if not self.solver:
            raise RuntimeError('No model loaded')
        return self.solver.solve_response_spectrum(spectrum, num_modes, combine) 