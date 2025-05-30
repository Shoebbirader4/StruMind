import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit, QHBoxLayout, QListWidget, QFileDialog, QMessageBox, QStatusBar, QToolTip, QCheckBox, QComboBox, QDialog, QFormLayout, QLineEdit, QInputDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from engine.core.api.engine_api import EngineAPI
from engine.core.geometry.analytical_model import AnalyticalModel, AnalyticalElement
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import copy
from core.section_library import add_custom_section

class OpenGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.displacements = None
        self.code_check_results = None
        self.selected_element = None
        self.setMinimumSize(400, 400)
        self.angle_x = 20
        self.angle_y = 30
        self.zoom = -8
        self.pan_x = 0
        self.pan_y = 0
        self.last_pos = None
        self.setMouseTracking(True)
        self.deformation_scale = 100  # For visualization
        self._viewport = (0, 0, 1, 1)
        self._proj_matrix = np.eye(4)
        self._model_matrix = np.eye(4)
        self.show_deformed = True
        self.show_code_check = True

    def update_model(self, model, displacements=None, code_check_results=None):
        self.model = model
        self.displacements = displacements
        self.code_check_results = code_check_results
        self.selected_element = None
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(1, 1, 1, 1)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/float(h or 1), 0.1, 100)
        glMatrixMode(GL_MODELVIEW)
        self._viewport = glGetIntegerv(GL_VIEWPORT)
        self._proj_matrix = glGetDoublev(GL_PROJECTION_MATRIX)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)
        self._model_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        if not self.model:
            return
        # Draw original geometry (gray)
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1)
        self._draw_elements(deformed=False)
        # Draw deformed shape (color-coded)
        if self.show_deformed and self.displacements is not None:
            self._draw_elements(deformed=True)

    def _draw_elements(self, deformed=False):
        for idx, elem in enumerate(getattr(self.model, 'elements', [])):
            color = self._get_element_color(idx) if self.show_code_check else (0.2, 0.2, 0.8)
            if self.selected_element == idx:
                color = (1.0, 0.7, 0.2)  # highlight selected
            glColor3f(*color)
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                n1, n2 = elem.nodes
                if deformed and self.displacements is not None:
                    n1 = self._apply_disp(n1, 0)
                    n2 = self._apply_disp(n2, 1)
                glLineWidth(5 if self.selected_element == idx else 2)
                glBegin(GL_LINES)
                glVertex3f(*self._to3d(n1))
                glVertex3f(*self._to3d(n2))
                glEnd()
            elif hasattr(elem, 'nodes') and len(elem.nodes) == 4:
                glBegin(GL_QUADS)
                for i, n in enumerate(elem.nodes):
                    node = n
                    if deformed and self.displacements is not None:
                        node = self._apply_disp(n, i)
                    glVertex3f(*self._to3d(node))
                glEnd()
                # Draw border for picking/selection
                if self.selected_element == idx:
                    glColor3f(1.0, 0.7, 0.2)
                    glLineWidth(3)
                    glBegin(GL_LINE_LOOP)
                    for n in elem.nodes:
                        glVertex3f(*self._to3d(n))
                    glEnd()
            elif hasattr(elem, 'nodes') and len(elem.nodes) == 8:
                edges = [
                    (0,1),(1,2),(2,3),(3,0), # bottom
                    (4,5),(5,6),(6,7),(7,4), # top
                    (0,4),(1,5),(2,6),(3,7)  # sides
                ]
                glBegin(GL_LINES)
                for i,j in edges:
                    node_i = elem.nodes[i]
                    node_j = elem.nodes[j]
                    if deformed and self.displacements is not None:
                        node_i = self._apply_disp(node_i, i)
                        node_j = self._apply_disp(node_j, j)
                    glVertex3f(*self._to3d(node_i))
                    glVertex3f(*self._to3d(node_j))
                glEnd()
                # Draw border for picking/selection
                if self.selected_element == idx:
                    glColor3f(1.0, 0.7, 0.2)
                    glLineWidth(3)
                    for face in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
                        glBegin(GL_LINE_LOOP)
                        for i in face:
                            glVertex3f(*self._to3d(elem.nodes[i]))
                        glEnd()

    def _get_element_color(self, idx):
        if self.code_check_results is not None and idx in self.code_check_results:
            result = self.code_check_results[idx]
            if all(result.get(k, True) for k in result):
                return (0.2, 0.8, 0.2)  # green
            else:
                return (0.9, 0.2, 0.2)  # red
        return (0.2, 0.2, 0.8)  # blue default

    def _to3d(self, n):
        if len(n) == 2:
            return (n[0], n[1], 0)
        return n

    def _apply_disp(self, n, idx):
        if self.displacements is None:
            return n
        node_idx = idx
        if len(self.displacements) >= (node_idx+1)*2:
            dx = self.displacements[node_idx*2] * self.deformation_scale
            dy = self.displacements[node_idx*2+1] * self.deformation_scale
            if len(n) == 2:
                return (n[0]+dx, n[1]+dy)
            elif len(n) == 3:
                return (n[0]+dx, n[1]+dy, n[2])
        return n

    def mousePressEvent(self, event):
        self.last_pos = (event.x(), event.y())
        self.last_btn = event.button()
        if event.button() == Qt.LeftButton:
            idx = self._pick_element(event.x(), event.y())
            if idx is not None:
                self.selected_element = idx
                self.parent().parent().show_element_properties(idx)
                self.update()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.x() - self.last_pos[0]
        dy = event.y() - self.last_pos[1]
        if event.buttons() & Qt.LeftButton:
            self.angle_x += dy
            self.angle_y += dx
        elif event.buttons() & Qt.RightButton:
            self.pan_x += dx * 0.01
            self.pan_y -= dy * 0.01
        self.last_pos = (event.x(), event.y())
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom += delta * 0.5
        self.update()

    def _pick_element(self, x, y):
        # Improved picking: find nearest element (beam, shell, solid) to mouse
        if not self.model or not hasattr(self.model, 'elements'):
            return None
        min_dist = float('inf')
        picked_idx = None
        for idx, elem in enumerate(self.model.elements):
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                n1, n2 = elem.nodes
                p1 = self._project(n1)
                p2 = self._project(n2)
                dist = self._point_line_distance((x, y), p1, p2)
                if dist < min_dist:
                    min_dist = dist
                    picked_idx = idx
            elif hasattr(elem, 'nodes') and len(elem.nodes) == 4:
                # Check if mouse is near quad face
                screen_pts = [self._project(n) for n in elem.nodes]
                if self._point_in_quad((x, y), screen_pts):
                    return idx
            elif hasattr(elem, 'nodes') and len(elem.nodes) == 8:
                # Check if mouse is near any face of the cube
                faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
                for face in faces:
                    screen_pts = [self._project(elem.nodes[i]) for i in face]
                    if self._point_in_quad((x, y), screen_pts):
                        return idx
        if min_dist < 15:
            return picked_idx
        return None

    def _project(self, n):
        model = self._model_matrix
        proj = self._proj_matrix
        viewport = self._viewport
        x, y, z = self._to3d(n)
        win = gluProject(x, y, z, model, proj, viewport)
        return (win[0], self.height() - win[1])

    def _point_line_distance(self, p, a, b):
        a = np.array(a)
        b = np.array(b)
        p = np.array(p)
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def _point_in_quad(self, p, quad):
        # Check if point p is inside the quad (screen space)
        def sign(o, a, b):
            return (o[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (o[1] - b[1])
        p1, p2, p3, p4 = quad
        # Split quad into two triangles
        in_tri1 = self._point_in_triangle(p, p1, p2, p3)
        in_tri2 = self._point_in_triangle(p, p1, p3, p4)
        return in_tri1 or in_tri2

    def _point_in_triangle(self, p, a, b, c):
        # Barycentric technique
        v0 = np.array(c) - np.array(a)
        v1 = np.array(b) - np.array(a)
        v2 = np.array(p) - np.array(a)
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return (u >= 0) and (v >= 0) and (u + v < 1)

class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    def plot_beam_diagram(self, diagram, diagram_type='M'):
        self.ax.clear()
        x = diagram['x']
        y = diagram[diagram_type]
        self.ax.plot(x, y, label=diagram_type)
        self.ax.set_title(f'{diagram_type} Diagram')
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel(diagram_type)
        self.ax.legend()
        self.canvas.draw()
    def plot_shell_contour(self, contour):
        self.ax.clear()
        disp = contour['disp']
        c = self.ax.tripcolor([i for i in range(len(disp))], [0]*len(disp), disp, shading='flat')
        self.figure.colorbar(c, ax=self.ax)
        self.ax.set_title('Shell Displacement Contour')
        self.canvas.draw()

class SectionMaterialDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle('Select Section and Material')
        layout = QFormLayout()
        self.section_cb = QComboBox()
        self.section_cb.addItems(self.api.get_section_names())
        self.material_cb = QComboBox()
        self.material_cb.addItems(self.api.get_material_names())
        layout.addRow('Section:', self.section_cb)
        layout.addRow('Material:', self.material_cb)
        self.setLayout(layout)
        self.selected_section = None
        self.selected_material = None
        self.section_cb.currentTextChanged.connect(self.update_selection)
        self.material_cb.currentTextChanged.connect(self.update_selection)
        self.update_selection()
    def update_selection(self):
        self.selected_section = self.section_cb.currentText()
        self.selected_material = self.material_cb.currentText()

class LoadCaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Define Load Case')
        layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.type_cb = QComboBox()
        self.type_cb.addItems(['point', 'distributed', 'area', 'temperature'])
        self.mag_edit = QLineEdit()
        self.loc_edit = QLineEdit()
        self.dir_edit = QLineEdit()
        self.span_edit = QLineEdit()
        self.area_edit = QLineEdit()
        self.temp_edit = QLineEdit()
        layout.addRow('Name:', self.name_edit)
        layout.addRow('Type:', self.type_cb)
        layout.addRow('Magnitude:', self.mag_edit)
        layout.addRow('Location (node/element idx):', self.loc_edit)
        layout.addRow('Direction (0=x,1=y,2=z):', self.dir_edit)
        layout.addRow('Span (start,end) [for distributed]:', self.span_edit)
        layout.addRow('Area (element,value) [for area]:', self.area_edit)
        layout.addRow('Temperature ΔT [for temp]:', self.temp_edit)
        self.setLayout(layout)
        self.type_cb.currentTextChanged.connect(self.update_fields)
        self.update_fields()
    def update_fields(self):
        t = self.type_cb.currentText()
        self.span_edit.setVisible(t == 'distributed')
        self.area_edit.setVisible(t == 'area')
        self.temp_edit.setVisible(t == 'temperature')
    def get_load(self):
        t = self.type_cb.currentText()
        span = tuple(map(float, self.span_edit.text().split(','))) if self.span_edit.text() and t == 'distributed' else None
        area = tuple(self.area_edit.text().split(',')) if self.area_edit.text() and t == 'area' else None
        temp = float(self.temp_edit.text()) if self.temp_edit.text() and t == 'temperature' else None
        return {
            'name': self.name_edit.text(),
            'type': t,
            'magnitude': float(self.mag_edit.text()),
            'location': int(self.loc_edit.text()),
            'direction': int(self.dir_edit.text()) if self.dir_edit.text() else 0,
            'span': span,
            'area': area,
            'temperature': temp
        }

class NonlinearAnalysisDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Nonlinear Analysis Options')
        layout = QFormLayout()
        self.steps_edit = QLineEdit('10')
        self.max_iter_edit = QLineEdit('20')
        self.tol_edit = QLineEdit('1e-4')
        layout.addRow('Load Steps:', self.steps_edit)
        layout.addRow('Max Iterations:', self.max_iter_edit)
        layout.addRow('Tolerance:', self.tol_edit)
        self.setLayout(layout)
    def get_options(self):
        return {
            'steps': int(self.steps_edit.text()),
            'max_iter': int(self.max_iter_edit.text()),
            'tol': float(self.tol_edit.text())
        }

class DrawingExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Drawing Options')
        layout = QFormLayout()
        self.view_cb = QComboBox()
        self.view_cb.addItems(['plan', 'elevation'])
        self.overlay_cb = QCheckBox('Overlay Force Diagrams')
        self.overlay_cb.setChecked(True)
        self.format_cb = QComboBox()
        self.format_cb.addItems(['SVG', 'PDF'])
        layout.addRow('View:', self.view_cb)
        layout.addRow('', self.overlay_cb)
        layout.addRow('Format:', self.format_cb)
        self.setLayout(layout)
    def get_options(self):
        return {
            'view': self.view_cb.currentText(),
            'overlay': self.overlay_cb.isChecked(),
            'format': self.format_cb.currentText()
        }

class ReportExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Report Options')
        layout = QFormLayout()
        self.format_cb = QComboBox()
        self.format_cb.addItems(['CSV', 'PDF', 'Excel'])
        layout.addRow('Format:', self.format_cb)
        self.setLayout(layout)
    def get_options(self):
        return {
            'format': self.format_cb.currentText()
        }

class StagedConstructionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Staged Construction Options')
        layout = QFormLayout()
        self.stages_edit = QLineEdit('3')
        layout.addRow('Number of Stages:', self.stages_edit)
        self.setLayout(layout)
    def get_options(self):
        return {
            'num_stages': int(self.stages_edit.text())
        }

class PushoverDialog(QDialog):
    def __init__(self, parent=None, num_nodes=1):
        super().__init__(parent)
        self.setWindowTitle('Pushover Analysis Options')
        layout = QFormLayout()
        self.node_edit = QLineEdit('0')
        self.dir_cb = QComboBox()
        self.dir_cb.addItems(['X', 'Y', 'Z'])
        self.steps_edit = QLineEdit('10')
        self.max_disp_edit = QLineEdit('0.1')
        layout.addRow('Control Node Index:', self.node_edit)
        layout.addRow('Direction:', self.dir_cb)
        layout.addRow('Steps:', self.steps_edit)
        layout.addRow('Max Displacement:', self.max_disp_edit)
        self.setLayout(layout)
    def get_options(self):
        return {
            'control_node': int(self.node_edit.text()),
            'direction': self.dir_cb.currentIndex(),
            'steps': int(self.steps_edit.text()),
            'max_disp': float(self.max_disp_edit.text())
        }

class TimeHistoryDialog(QDialog):
    def __init__(self, parent=None, n_dof=1):
        super().__init__(parent)
        self.setWindowTitle('Time History Analysis Options')
        layout = QFormLayout()
        self.dt_edit = QLineEdit('0.01')
        self.t_total_edit = QLineEdit('1.0')
        self.dof_edit = QLineEdit('0')
        self.amp_edit = QLineEdit('1000')
        layout.addRow('Time Step (dt):', self.dt_edit)
        layout.addRow('Total Time:', self.t_total_edit)
        layout.addRow('DOF to Load (index):', self.dof_edit)
        layout.addRow('Load Amplitude:', self.amp_edit)
        self.setLayout(layout)
    def get_options(self):
        return {
            'dt': float(self.dt_edit.text()),
            't_total': float(self.t_total_edit.text()),
            'dof': int(self.dof_edit.text()),
            'amp': float(self.amp_edit.text())
        }

class ResponseSpectrumDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Response Spectrum Analysis Options')
        layout = QFormLayout()
        self.num_modes_edit = QLineEdit('3')
        layout.addRow('Number of Modes:', self.num_modes_edit)
        self.setLayout(layout)
    def get_options(self):
        return {
            'num_modes': int(self.num_modes_edit.text())
        }

class CustomSectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Create/Edit Custom Section')
        layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.shape_cb = QComboBox()
        self.shape_cb.addItems(['rect', 'circle', 'polygon'])
        self.param1_edit = QLineEdit()
        self.param2_edit = QLineEdit()
        self.polygon_edit = QLineEdit()
        layout.addRow('Section Name:', self.name_edit)
        layout.addRow('Shape:', self.shape_cb)
        layout.addRow('b (width) or r (radius):', self.param1_edit)
        layout.addRow('h (height):', self.param2_edit)
        layout.addRow('Polygon (x1 y1, x2 y2, ...):', self.polygon_edit)
        self.setLayout(layout)
        self.shape_cb.currentTextChanged.connect(self.update_fields)
        self.update_fields()
    def update_fields(self):
        shape = self.shape_cb.currentText()
        self.param1_edit.setVisible(shape in ['rect', 'circle'])
        self.param2_edit.setVisible(shape == 'rect')
        self.polygon_edit.setVisible(shape == 'polygon')
    def get_section(self):
        name = self.name_edit.text()
        shape = self.shape_cb.currentText()
        if shape == 'rect':
            b = float(self.param1_edit.text())
            h = float(self.param2_edit.text())
            params = {'shape': 'rect', 'b': b, 'h': h}
            return name, params, None
        elif shape == 'circle':
            r = float(self.param1_edit.text())
            params = {'shape': 'circle', 'r': r}
            return name, params, None
        elif shape == 'polygon':
            pts = self.polygon_edit.text().strip().split(',')
            polygon = [tuple(map(float, pt.strip().split())) for pt in pts if pt.strip()]
            return name, {}, polygon
        return name, {}, None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Structural Engine GUI')
        self.setGeometry(100, 100, 1200, 700)
        self.api = EngineAPI()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        # Sidebar for model input
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.load_model_btn = QPushButton('Load Demo Model')
        self.load_model_btn.setToolTip('Load a built-in demo model')
        self.load_model_btn.clicked.connect(self.load_demo_model)
        self.load_file_btn = QPushButton('Load Model from File')
        self.load_file_btn.setToolTip('Load a model from a JSON, CSV, or IFC file')
        self.load_file_btn.clicked.connect(self.load_model_from_file)
        self.section_mat_btn = QPushButton('Select Section/Material')
        self.section_mat_btn.setToolTip('Select section and material for new elements')
        self.section_mat_btn.clicked.connect(self.select_section_material)
        self.custom_section_btn = QPushButton('Create Custom Section')
        self.custom_section_btn.setToolTip('Create or edit a custom/parametric section')
        self.custom_section_btn.clicked.connect(self.create_custom_section)
        self.load_case_btn = QPushButton('Define Load Case')
        self.load_case_btn.setToolTip('Define a new load case')
        self.load_case_btn.clicked.connect(self.define_load_case)
        self.export_btn = QPushButton('Export Results')
        self.export_btn.setToolTip('Export analysis results and code checks to CSV')
        self.export_btn.clicked.connect(self.export_results)
        self.zoom_fit_btn = QPushButton('Zoom to Fit')
        self.zoom_fit_btn.setToolTip('Zoom to fit the model in the view')
        self.zoom_fit_btn.clicked.connect(self.zoom_to_fit)
        self.reset_view_btn = QPushButton('Reset View')
        self.reset_view_btn.setToolTip('Reset camera to default view')
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.show_deformed_cb = QCheckBox('Show Deformed Shape')
        self.show_deformed_cb.setChecked(True)
        self.show_deformed_cb.setToolTip('Toggle display of deformed shape')
        self.show_deformed_cb.stateChanged.connect(self.toggle_deformed)
        self.show_code_check_cb = QCheckBox('Color by Code Check')
        self.show_code_check_cb.setChecked(True)
        self.show_code_check_cb.setToolTip('Toggle color coding by code check results')
        self.show_code_check_cb.stateChanged.connect(self.toggle_code_check)
        self.run_analysis_btn = QPushButton('Run Analysis')
        self.run_analysis_btn.setToolTip('Run structural analysis')
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_checks_btn = QPushButton('Run Code Checks')
        self.run_checks_btn.setToolTip('Run code checks for all elements')
        self.run_checks_btn.clicked.connect(self.run_code_checks)
        self.nonlinear_btn = QPushButton('Nonlinear Analysis')
        self.nonlinear_btn.setToolTip('Run nonlinear analysis (stub)')
        self.nonlinear_btn.clicked.connect(self.run_nonlinear)
        self.dynamic_btn = QPushButton('Dynamic Analysis')
        self.dynamic_btn.setToolTip('Run dynamic analysis (stub)')
        self.dynamic_btn.clicked.connect(self.run_dynamic)
        self.buckling_btn = QPushButton('Buckling Analysis')
        self.buckling_btn.setToolTip('Run buckling analysis (stub)')
        self.buckling_btn.clicked.connect(self.run_buckling)
        self.staged_btn = QPushButton('Staged Construction')
        self.staged_btn.setToolTip('Run staged construction analysis')
        self.staged_btn.clicked.connect(self.staged_construction)
        self.pushover_btn = QPushButton('Pushover Analysis')
        self.pushover_btn.setToolTip('Run pushover (capacity) analysis')
        self.pushover_btn.clicked.connect(self.pushover_analysis)
        self.time_history_btn = QPushButton('Time History Analysis')
        self.time_history_btn.setToolTip('Run time history (direct integration) analysis')
        self.time_history_btn.clicked.connect(self.time_history_analysis)
        self.response_spectrum_btn = QPushButton('Response Spectrum Analysis')
        self.response_spectrum_btn.setToolTip('Run response spectrum analysis')
        self.response_spectrum_btn.clicked.connect(self.response_spectrum_analysis)
        self.add_beam_btn = QPushButton('Add Beam')
        self.add_beam_btn.setToolTip('Add a new beam element with selected section/material')
        self.add_beam_btn.clicked.connect(self.add_beam)
        self.add_load_btn = QPushButton('Add Load')
        self.add_load_btn.setToolTip('Add a new load to the model')
        self.add_load_btn.clicked.connect(self.add_load)
        self.add_column_btn = QPushButton('Add Column')
        self.add_column_btn.setToolTip('Add a new column element with selected section/material')
        self.add_column_btn.clicked.connect(self.add_column)
        self.add_shell_btn = QPushButton('Add Shell')
        self.add_shell_btn.setToolTip('Add a new shell element (4 nodes)')
        self.add_shell_btn.clicked.connect(self.add_shell)
        self.add_solid_btn = QPushButton('Add Solid')
        self.add_solid_btn.setToolTip('Add a new solid element (8 nodes)')
        self.add_solid_btn.clicked.connect(self.add_solid)
        self.add_cable_btn = QPushButton('Add Cable')
        self.add_cable_btn.setToolTip('Add a new cable element with selected section/material')
        self.add_cable_btn.clicked.connect(self.add_cable)
        self.modal_btn = QPushButton('Modal Analysis')
        self.modal_btn.setToolTip('Run modal (eigenvalue) analysis')
        self.modal_btn.clicked.connect(self.run_modal)
        self.ifc_import_btn = QPushButton('Import IFC')
        self.ifc_import_btn.setToolTip('Import model from IFC file')
        self.ifc_import_btn.clicked.connect(self.import_ifc)
        self.ifc_export_btn = QPushButton('Export IFC')
        self.ifc_export_btn.setToolTip('Export model to IFC file')
        self.ifc_export_btn.clicked.connect(self.export_ifc)
        self.drawings_btn = QPushButton('Generate Drawings')
        self.drawings_btn.setToolTip('Generate construction/shop drawings')
        self.drawings_btn.clicked.connect(self.generate_drawings)
        self.report_btn = QPushButton('Generate Report')
        self.report_btn.setToolTip('Generate analysis/design report')
        self.report_btn.clicked.connect(self.generate_report)
        self.undo_btn = QPushButton('Undo')
        self.undo_btn.setToolTip('Undo last action')
        self.undo_btn.clicked.connect(self.undo_action)
        self.redo_btn = QPushButton('Redo')
        self.redo_btn.setToolTip('Redo last undone action')
        self.redo_btn.clicked.connect(self.redo_action)
        self.element_list = QListWidget()
        self.sidebar_layout.addWidget(QLabel('Model Input'))
        self.sidebar_layout.addWidget(self.load_model_btn)
        self.sidebar_layout.addWidget(self.load_file_btn)
        self.sidebar_layout.addWidget(self.section_mat_btn)
        self.sidebar_layout.addWidget(self.custom_section_btn)
        self.sidebar_layout.addWidget(self.load_case_btn)
        self.sidebar_layout.addWidget(self.export_btn)
        self.sidebar_layout.addWidget(self.zoom_fit_btn)
        self.sidebar_layout.addWidget(self.reset_view_btn)
        self.sidebar_layout.addWidget(self.show_deformed_cb)
        self.sidebar_layout.addWidget(self.show_code_check_cb)
        self.sidebar_layout.addWidget(self.run_analysis_btn)
        self.sidebar_layout.addWidget(self.run_checks_btn)
        self.sidebar_layout.addWidget(self.nonlinear_btn)
        self.sidebar_layout.addWidget(self.dynamic_btn)
        self.sidebar_layout.addWidget(self.buckling_btn)
        self.sidebar_layout.addWidget(self.staged_btn)
        self.sidebar_layout.addWidget(self.pushover_btn)
        self.sidebar_layout.addWidget(self.time_history_btn)
        self.sidebar_layout.addWidget(self.response_spectrum_btn)
        self.sidebar_layout.addWidget(self.add_beam_btn)
        self.sidebar_layout.addWidget(self.add_load_btn)
        self.sidebar_layout.addWidget(self.add_column_btn)
        self.sidebar_layout.addWidget(self.add_shell_btn)
        self.sidebar_layout.addWidget(self.add_solid_btn)
        self.sidebar_layout.addWidget(self.add_cable_btn)
        self.sidebar_layout.addWidget(self.modal_btn)
        self.sidebar_layout.addWidget(self.ifc_import_btn)
        self.sidebar_layout.addWidget(self.ifc_export_btn)
        self.sidebar_layout.addWidget(self.drawings_btn)
        self.sidebar_layout.addWidget(self.report_btn)
        self.sidebar_layout.addWidget(self.undo_btn)
        self.sidebar_layout.addWidget(self.redo_btn)
        self.sidebar_layout.addWidget(QLabel('Elements'))
        self.sidebar_layout.addWidget(self.element_list)
        self.sidebar.setLayout(self.sidebar_layout)
        # Results panel
        self.results_panel = QTextEdit()
        self.results_panel.setReadOnly(True)
        self.results_panel.setMinimumWidth(200)
        self.results_panel.setText('Results will be shown here.')
        # OpenGL visualization widget
        from engine.gui.qt_main import OpenGLWidget  # Avoid circular import
        self.opengl_widget = OpenGLWidget(self)
        # Plot panel for diagrams/contours
        self.plot_panel = PlotPanel(self)
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.opengl_widget, stretch=2)
        main_layout.addWidget(self.plot_panel, stretch=1)
        main_layout.addWidget(self.results_panel)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        QToolTip.setFont(self.font())
        self.selected_section = None
        self.selected_material = None
        self.undo_stack = []
        self.redo_stack = []
    def select_section_material(self):
        dlg = SectionMaterialDialog(self.api, self)
        if dlg.exec_():
            self.selected_section = dlg.selected_section
            self.selected_material = dlg.selected_material
            self.status_bar.showMessage(f'Selected section: {dlg.selected_section}, material: {dlg.selected_material}')
    def create_custom_section(self):
        dlg = CustomSectionDialog(self)
        if dlg.exec_():
            name, params, polygon = dlg.get_section()
            add_custom_section(name, params, polygon)
            # Update section combo box in SectionMaterialDialog if open
            if hasattr(self, 'section_mat_btn'):
                # This will only update for new dialogs, not already open ones
                pass
            self.status_bar.showMessage(f'Custom section "{name}" added.')
    def define_load_case(self):
        dlg = LoadCaseDialog(self)
        if dlg.exec_():
            load = dlg.get_load()
            self.status_bar.showMessage(f'Defined load case: {load}')
    def run_nonlinear(self):
        dlg = NonlinearAnalysisDialog(self)
        if dlg.exec_():
            opts = dlg.get_options()
            try:
                result = self.api.run_nonlinear_analysis(opts)
                self.results_panel.setText(f'Nonlinear analysis result:\nConverged: {result["converged"]}\nSteps: {result["steps"]}\nTolerance: {result["tol"]}')
                self.status_bar.showMessage('Nonlinear analysis complete.')
                # Plot load vs. tip displacement curve
                u_hist = result['displacement_history']
                steps = result['steps']
                tip_node = len(self.api.model.nodes) - 1
                tip_disp = [u[tip_node*3+1] for u in u_hist]  # y-displacement
                loads = [(i+1)/steps for i in range(len(u_hist))]  # normalized load factor
                self.plot_panel.ax.clear()
                self.plot_panel.ax.plot(tip_disp, loads, marker='o')
                self.plot_panel.ax.set_xlabel('Tip Displacement (y)')
                self.plot_panel.ax.set_ylabel('Normalized Load')
                self.plot_panel.ax.set_title('Load-Displacement Curve (Tip)')
                self.plot_panel.ax.grid(True)
                self.plot_panel.canvas.draw()
            except Exception as e:
                self.results_panel.setText(f'Nonlinear analysis error: {e}')
                self.status_bar.showMessage('Nonlinear analysis error.')
    def run_dynamic(self):
        self.status_bar.showMessage('Dynamic analysis (stub)')
        self.results_panel.setText('Dynamic analysis is a stub.')
    def run_buckling(self):
        try:
            result = self.api.run_buckling_analysis()
            crit_lambda = result['critical_load_factor']
            crit_mode = result['mode_shape']
            if crit_lambda is not None:
                self.results_panel.setText(f'Buckling analysis complete.\nCritical load factor: {crit_lambda:.4g}')
                self.status_bar.showMessage('Buckling analysis complete.')
                # Plot mode shape if available
                if crit_mode is not None:
                    self.plot_panel.ax.clear()
                    # Plot undeformed structure
                    nodes = self.api.model.nodes
                    for elem in self.api.model.elements:
                        if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                            n1, n2 = elem.nodes
                            x = [n1[0], n2[0]]
                            y = [n1[1], n2[1]]
                            self.plot_panel.ax.plot(x, y, 'k--', alpha=0.5)
                    # Plot deformed shape (mode)
                    scale = 10  # scale factor for visibility
                    for elem in self.api.model.elements:
                        if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                            n1, n2 = elem.nodes
                            idx1 = self.api.model.nodes.index(n1)
                            idx2 = self.api.model.nodes.index(n2)
                            u1 = crit_mode[idx1*3] if len(crit_mode) > idx1*3 else 0
                            v1 = crit_mode[idx1*3+1] if len(crit_mode) > idx1*3+1 else 0
                            u2 = crit_mode[idx2*3] if len(crit_mode) > idx2*3 else 0
                            v2 = crit_mode[idx2*3+1] if len(crit_mode) > idx2*3+1 else 0
                            x_def = [n1[0] + scale*u1, n2[0] + scale*u2]
                            y_def = [n1[1] + scale*v1, n2[1] + scale*v2]
                            self.plot_panel.ax.plot(x_def, y_def, 'r-', lw=2, label='Buckling mode' if elem==self.api.model.elements[0] else None)
                    self.plot_panel.ax.set_title('Buckling Mode Shape')
                    self.plot_panel.ax.set_aspect('equal')
                    self.plot_panel.ax.legend()
                    self.plot_panel.canvas.draw()
            else:
                self.results_panel.setText('Buckling analysis failed or not available.')
                self.status_bar.showMessage('Buckling analysis failed.')
        except Exception as e:
            self.results_panel.setText(f'Buckling analysis error: {e}')
            self.status_bar.showMessage('Buckling analysis error.')
    def show_element_properties(self, idx):
        if self.api.model and hasattr(self.api.model, 'elements') and idx < len(self.api.model.elements):
            elem = self.api.model.elements[idx]
            props = f'Element {idx+1} properties:\nNodes: {elem.nodes}\nSection: {getattr(elem, "section", {})}\nMaterial: {getattr(elem, "material", {})}'
            self.results_panel.setText(props)
            self.status_bar.showMessage(f'Element {idx+1} selected.')
            # Show diagrams/contours if available
            diagrams = self.api.get_beam_internal_force_diagrams()
            if diagrams and idx < len(diagrams):
                self.plot_panel.plot_beam_diagram(diagrams[idx], diagram_type='M')
            contours = self.api.get_shell_displacement_contours()
            if contours and idx < len(contours):
                self.plot_panel.plot_shell_contour(contours[idx])
    def toggle_deformed(self, state):
        self.opengl_widget.show_deformed = bool(state)
        self.opengl_widget.update()
    def toggle_code_check(self, state):
        self.opengl_widget.show_code_check = bool(state)
        self.opengl_widget.update()
    def load_demo_model(self):
        node1 = (0, 0)
        node2 = (5, 0)
        node3 = (5, 3)
        section = {'A': 0.01, 'I': 1e-6, 'Z': 1e-4}
        material = {'E': 210e9, 'Fy': 250}
        class DemoElement:
            def __init__(self, nodes):
                self.nodes = nodes
                self.section = section
                self.material = material
                def length(self):
                    return 5.0
                self.length = length.__get__(self)
        elem1 = DemoElement([node1, node2])
        elem2 = DemoElement([node2, node3])
        model = AnalyticalModel()
        model.nodes = [node1, node2, node3]
        model.elements = [elem1, elem2]
        self.api.load_model(model)
        self.push_undo()
        self.element_list.clear()
        for i, elem in enumerate(model.elements):
            self.element_list.addItem(f'Element {i+1}: Nodes {elem.nodes}')
        self.opengl_widget.update_model(model)
        self.results_panel.setText('Model loaded.')
        self.status_bar.showMessage('Model loaded.')
    def load_model_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Model File', '', 'JSON/CSV/IFC Files (*.json *.csv *.ifc);;All Files (*)')
        if not fname:
            return
        try:
            if fname.endswith('.json'):
                import json
                with open(fname, 'r') as f:
                    data = json.load(f)
                nodes = [tuple(n) for n in data['nodes']]
                section = data['section']
                material = data['material']
                class FileElement:
                    def __init__(self, nodes):
                        self.nodes = nodes
                        self.section = section
                        self.material = material
                        def length(self):
                            return np.linalg.norm(np.array(nodes[1]) - np.array(nodes[0]))
                        self.length = length.__get__(self)
                elements = [FileElement([tuple(n) for n in elem['nodes']]) for elem in data['elements']]
                model = AnalyticalModel()
                model.nodes = nodes
                model.elements = elements
            elif fname.endswith('.csv'):
                import csv
                nodes = []
                elements = []
                section = {'A': 0.01, 'I': 1e-6, 'Z': 1e-4}
                material = {'E': 210e9, 'Fy': 250}
                with open(fname, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[0] == 'node':
                            nodes.append(tuple(map(float, row[1:])))
                        elif row[0] == 'element':
                            nidx = list(map(int, row[1:]))
                            class CsvElement:
                                def __init__(self, nodes):
                                    self.nodes = nodes
                                    self.section = section
                                    self.material = material
                                    def length(self):
                                        return np.linalg.norm(np.array(nodes[1]) - np.array(nodes[0]))
                                    self.length = length.__get__(self)
                            elements.append(CsvElement([nodes[nidx[0]], nodes[nidx[1]]]))
                model = AnalyticalModel()
                model.nodes = nodes
                model.elements = elements
            elif fname.endswith('.ifc'):
                # IFC import stub
                QMessageBox.information(self, 'IFC Import', 'IFC import is a stub. (Use IfcOpenShell for real import)')
                return
            else:
                raise Exception('Unsupported file format')
            self.api.load_model(model)
            self.push_undo()
            self.element_list.clear()
            for i, elem in enumerate(model.elements):
                self.element_list.addItem(f'Element {i+1}: Nodes {elem.nodes}')
            self.opengl_widget.update_model(model)
            self.results_panel.setText('Model loaded from file.')
            self.status_bar.showMessage('Model loaded from file.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model: {e}')
            self.status_bar.showMessage('Failed to load model.')
    def export_results(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Export Results', '', 'CSV Files (*.csv);;All Files (*)')
        if not fname:
            return
        try:
            with open(fname, 'w') as f:
                f.write('Element,Displacement,CodeCheck\n')
                for i, elem in enumerate(self.api.model.elements):
                    disp = self.api.results['displacements'][i] if self.api.results and 'displacements' in self.api.results else ''
                    code = self.api.code_check_results[i] if self.api.code_check_results and i in self.api.code_check_results else ''
                    f.write(f'{i+1},{disp},{code}\n')
            self.status_bar.showMessage('Results exported.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to export results: {e}')
            self.status_bar.showMessage('Failed to export results.')
    def zoom_to_fit(self):
        # Simple zoom to fit (reset view)
        self.opengl_widget.angle_x = 20
        self.opengl_widget.angle_y = 30
        self.opengl_widget.zoom = -8
        self.opengl_widget.pan_x = 0
        self.opengl_widget.pan_y = 0
        self.opengl_widget.update()
        self.status_bar.showMessage('Zoomed to fit.')
    def reset_view(self):
        self.zoom_to_fit()
        self.status_bar.showMessage('View reset.')
    def run_analysis(self):
        try:
            class DummyLoad:
                def __init__(self):
                    self.location = 1
                    self.direction = 1
                    self.magnitude = -10000.0
            loads = [DummyLoad()]
            results = self.api.run_analysis(loads)
            disp = results['displacements']
            self.results_panel.setText(f'Analysis complete.\nDisplacements: {disp}')
            self.opengl_widget.update_model(self.api.model, displacements=disp)
            self.status_bar.showMessage('Analysis complete.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Analysis failed: {e}')
            self.status_bar.showMessage('Analysis failed.')
    def run_code_checks(self):
        try:
            forces_by_element = {
                0: {'M': 200000, 'V': 10000, 'N': 50000, 'L': 5.0, 'Deflection': 0.01},
                1: {'M': 150000, 'V': 8000, 'N': 40000, 'L': 5.0, 'Deflection': 0.008},
            }
            results = self.api.run_code_checks('AISC', forces_by_element)
            self.results_panel.setText(f'Code checks:\n{results}')
            self.opengl_widget.update_model(self.api.model, code_check_results=results)
            self.status_bar.showMessage('Code checks complete.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Code checks failed: {e}')
            self.status_bar.showMessage('Code checks failed.')
    def push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.api.model))
        self.redo_stack.clear()
        self.update_undo_redo_buttons()
    def restore_model(self, model):
        self.api.load_model(model)
        self.element_list.clear()
        for i, elem in enumerate(model.elements):
            if hasattr(elem, 'nodes'):
                self.element_list.addItem(f'Element {i+1}: Nodes {elem.nodes}')
            else:
                self.element_list.addItem(f'Element {i+1}')
        self.opengl_widget.update_model(model)
    def undo_action(self):
        try:
            if len(self.undo_stack) < 2:
                self.status_bar.showMessage('Nothing to undo.')
                return
            self.redo_stack.append(self.undo_stack.pop())
            model = copy.deepcopy(self.undo_stack[-1])
            self.restore_model(model)
            self.status_bar.showMessage('Undo complete.')
            self.update_undo_redo_buttons()
        except Exception as e:
            self.results_panel.setText(f'Undo error: {e}')
            self.status_bar.showMessage('Undo error.')
    def redo_action(self):
        try:
            if not self.redo_stack:
                self.status_bar.showMessage('Nothing to redo.')
                return
            model = self.redo_stack.pop()
            self.undo_stack.append(copy.deepcopy(model))
            self.restore_model(model)
            self.status_bar.showMessage('Redo complete.')
            self.update_undo_redo_buttons()
        except Exception as e:
            self.results_panel.setText(f'Redo error: {e}')
            self.status_bar.showMessage('Redo error.')
    def update_undo_redo_buttons(self):
        self.undo_btn.setEnabled(len(self.undo_stack) > 1)
        self.redo_btn.setEnabled(bool(self.redo_stack))
    def add_beam(self):
        # Prompt for node indices
        nodes = self.api.model.nodes if self.api.model else []
        if not nodes or self.selected_section is None or self.selected_material is None:
            self.status_bar.showMessage('Please load a model and select section/material first.')
            return
        idx1, ok1 = QInputDialog.getInt(self, 'Add Beam', 'Start node index (0-based):', 0, 0, len(nodes)-1)
        if not ok1:
            return
        idx2, ok2 = QInputDialog.getInt(self, 'Add Beam', 'End node index (0-based):', 1, 0, len(nodes)-1)
        if not ok2 or idx1 == idx2:
            return
        from engine.core.elements.beam import BeamElement
        section = self.api.get_section(self.selected_section)
        material = self.api.get_material(self.selected_material)
        elem = BeamElement(nodes[idx1], nodes[idx2], section, material)
        self.api.model.add_element(elem)
        self.push_undo()
        self.element_list.addItem(f'Element {len(self.api.model.elements)}: Nodes {elem.start_node, elem.end_node}')
        self.opengl_widget.update_model(self.api.model)
        self.status_bar.showMessage('Beam element added.')
    def add_load(self):
        dlg = LoadCaseDialog(self)
        if dlg.exec_():
            load = dlg.get_load()
            from core.analysis.loads import Load
            l = Load(load['type'], load['magnitude'], load['location'], load['direction'], load.get('span'), load.get('area'), load.get('temperature'))
            self.api.model.add_load(l)
            self.push_undo()
            self.status_bar.showMessage(f'Load added: {load}')
    def add_column(self):
        nodes = self.api.model.nodes if self.api.model else []
        if not nodes or self.selected_section is None or self.selected_material is None:
            self.status_bar.showMessage('Please load a model and select section/material first.')
            return
        idx1, ok1 = QInputDialog.getInt(self, 'Add Column', 'Base node index (0-based):', 0, 0, len(nodes)-1)
        if not ok1:
            return
        idx2, ok2 = QInputDialog.getInt(self, 'Add Column', 'Top node index (0-based):', 1, 0, len(nodes)-1)
        if not ok2 or idx1 == idx2:
            return
        from engine.core.elements.column import ColumnElement
        section = self.api.get_section(self.selected_section)
        material = self.api.get_material(self.selected_material)
        elem = ColumnElement(nodes[idx1], nodes[idx2], section, material)
        self.api.model.add_element(elem)
        self.push_undo()
        self.element_list.addItem(f'Column {len(self.api.model.elements)}: Nodes {elem.base_node, elem.top_node}')
        self.opengl_widget.update_model(self.api.model)
        self.status_bar.showMessage('Column element added.')
    def add_shell(self):
        nodes = self.api.model.nodes if self.api.model else []
        if len(nodes) < 4 or self.selected_section is None or self.selected_material is None:
            self.status_bar.showMessage('Need at least 4 nodes and section/material.')
            return
        idxs, ok = QInputDialog.getText(self, 'Add Shell', 'Node indices (comma-separated, 4 nodes):', text='0,1,2,3')
        if not ok:
            return
        idxs = [int(i.strip()) for i in idxs.split(',') if i.strip().isdigit()]
        if len(idxs) != 4:
            self.status_bar.showMessage('Please enter 4 valid node indices.')
            return
        from engine.core.elements.shell import ShellElement
        section = self.api.get_section(self.selected_section)
        material = self.api.get_material(self.selected_material)
        elem = ShellElement([nodes[i] for i in idxs], section, material)
        self.api.model.add_element(elem)
        self.push_undo()
        self.element_list.addItem(f'Shell {len(self.api.model.elements)}: Nodes {[nodes[i] for i in idxs]}')
        self.opengl_widget.update_model(self.api.model)
        self.status_bar.showMessage('Shell element added.')
    def add_solid(self):
        nodes = self.api.model.nodes if self.api.model else []
        if len(nodes) < 8 or self.selected_section is None or self.selected_material is None:
            self.status_bar.showMessage('Need at least 8 nodes and section/material.')
            return
        idxs, ok = QInputDialog.getText(self, 'Add Solid', 'Node indices (comma-separated, 8 nodes):', text='0,1,2,3,4,5,6,7')
        if not ok:
            return
        idxs = [int(i.strip()) for i in idxs.split(',') if i.strip().isdigit()]
        if len(idxs) != 8:
            self.status_bar.showMessage('Please enter 8 valid node indices.')
            return
        from engine.core.elements.solid import SolidElement
        section = self.api.get_section(self.selected_section)
        material = self.api.get_material(self.selected_material)
        elem = SolidElement([nodes[i] for i in idxs], section, material)
        self.api.model.add_element(elem)
        self.push_undo()
        self.element_list.addItem(f'Solid {len(self.api.model.elements)}: Nodes {[nodes[i] for i in idxs]}')
        self.opengl_widget.update_model(self.api.model)
        self.status_bar.showMessage('Solid element added.')
    def add_cable(self):
        nodes = self.api.model.nodes if self.api.model else []
        if not nodes or self.selected_section is None or self.selected_material is None:
            self.status_bar.showMessage('Please load a model and select section/material first.')
            return
        idx1, ok1 = QInputDialog.getInt(self, 'Add Cable', 'Start node index (0-based):', 0, 0, len(nodes)-1)
        if not ok1:
            return
        idx2, ok2 = QInputDialog.getInt(self, 'Add Cable', 'End node index (0-based):', 1, 0, len(nodes)-1)
        if not ok2 or idx1 == idx2:
            return
        from engine.core.elements.cable import CableElement
        section = self.api.get_section(self.selected_section)
        material = self.api.get_material(self.selected_material)
        elem = CableElement(nodes[idx1], nodes[idx2], section, material)
        self.api.model.add_element(elem)
        self.push_undo()
        self.element_list.addItem(f'Cable {len(self.api.model.elements)}: Nodes {elem.start_node, elem.end_node}')
        self.opengl_widget.update_model(self.api.model)
        self.status_bar.showMessage('Cable element added.')
    def run_modal(self):
        try:
            result = self.api.run_modal_analysis()
            freqs = result['frequencies']
            mode_shapes = result['mode_shapes']
            if len(freqs) == 0:
                self.results_panel.setText('Modal analysis failed or not available.')
                self.status_bar.showMessage('Modal analysis failed.')
                return
            # Show frequency table
            freq_table = 'Mode\tFrequency (Hz)\n' + '\n'.join([f'{i+1}\t{freqs[i]:.4f}' for i in range(len(freqs))])
            self.results_panel.setText(f'Modal analysis result:\n{freq_table}\nSelect a mode to plot.')
            self.status_bar.showMessage('Modal analysis complete.')
            # Add mode selection and plot
            from PyQt5.QtWidgets import QComboBox
            if not hasattr(self, 'mode_cb'):
                self.mode_cb = QComboBox(self)
                self.plot_panel.layout().addWidget(self.mode_cb)
                self.mode_cb.currentIndexChanged.connect(self.plot_selected_mode_shape)
            self.mode_cb.clear()
            for i in range(len(freqs)):
                self.mode_cb.addItem(f'Mode {i+1}')
            self._modal_mode_shapes = mode_shapes
            self._modal_freqs = freqs
            self.plot_selected_mode_shape(0)
        except Exception as e:
            self.results_panel.setText(f'Modal analysis error: {e}')
            self.status_bar.showMessage('Modal analysis error.')
    def plot_selected_mode_shape(self, idx):
        if not hasattr(self, '_modal_mode_shapes') or self._modal_mode_shapes is None:
            return
        mode_shape = self._modal_mode_shapes[:, idx]
        nodes = self.api.model.nodes
        scale = 10  # for visibility
        self.plot_panel.ax.clear()
        # Plot undeformed structure
        for elem in self.api.model.elements:
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                n1, n2 = elem.nodes
                x = [n1[0], n2[0]]
                y = [n1[1], n2[1]]
                self.plot_panel.ax.plot(x, y, 'k--', alpha=0.5)
        # Plot deformed shape (mode)
        for elem in self.api.model.elements:
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                n1, n2 = elem.nodes
                idx1 = nodes.index(n1)
                idx2 = nodes.index(n2)
                u1 = mode_shape[idx1*3] if len(mode_shape) > idx1*3 else 0
                v1 = mode_shape[idx1*3+1] if len(mode_shape) > idx1*3+1 else 0
                u2 = mode_shape[idx2*3] if len(mode_shape) > idx2*3 else 0
                v2 = mode_shape[idx2*3+1] if len(mode_shape) > idx2*3+1 else 0
                x_def = [n1[0] + scale*u1, n2[0] + scale*u2]
                y_def = [n1[1] + scale*v1, n2[1] + scale*v2]
                self.plot_panel.ax.plot(x_def, y_def, 'r-', lw=2, label='Mode shape' if elem==self.api.model.elements[0] else None)
        self.plot_panel.ax.set_title(f'Mode Shape {idx+1}')
        self.plot_panel.ax.set_aspect('equal')
        self.plot_panel.ax.legend()
        self.plot_panel.canvas.draw()
    def import_ifc(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Import IFC File', '', 'IFC Files (*.ifc);;All Files (*)')
            if not fname:
                return
            result = self.api.import_ifc(fname)
            self.results_panel.setText(result)
            self.status_bar.showMessage('IFC import complete.')
            self.element_list.clear()
            for i, elem in enumerate(self.api.model.elements):
                self.element_list.addItem(f'Element {i+1}: Nodes {elem.nodes}')
            self.opengl_widget.update_model(self.api.model)
        except Exception as e:
            self.results_panel.setText(f'IFC import error: {e}')
            self.status_bar.showMessage('IFC import error.')
    def export_ifc(self):
        try:
            fname, _ = QFileDialog.getSaveFileName(self, 'Export IFC File', '', 'IFC Files (*.ifc);;All Files (*)')
            if not fname:
                return
            result = self.api.export_ifc(fname)
            self.results_panel.setText(result)
            self.status_bar.showMessage('IFC export complete.')
        except Exception as e:
            self.results_panel.setText(f'IFC export error: {e}')
            self.status_bar.showMessage('IFC export error.')
    def generate_drawings(self):
        try:
            dlg = DrawingExportDialog(self)
            if not dlg.exec_():
                return
            opts = dlg.get_options()
            if opts['format'] == 'SVG':
                fname, _ = QFileDialog.getSaveFileName(self, 'Export Drawing', '', 'SVG Files (*.svg);;All Files (*)')
            else:
                fname, _ = QFileDialog.getSaveFileName(self, 'Export Drawing', '', 'PDF Files (*.pdf);;All Files (*)')
            if not fname:
                return
            result = self.api.generate_drawings(fname, view=opts['view'], overlay_forces=opts['overlay'], pdf=(opts['format']=='PDF'))
            self.results_panel.setText(result)
            self.status_bar.showMessage('Drawing export complete.')
        except Exception as e:
            self.results_panel.setText(f'Drawing generation error: {e}')
            self.status_bar.showMessage('Drawing generation error.')
    def generate_report(self):
        try:
            dlg = ReportExportDialog(self)
            if not dlg.exec_():
                return
            opts = dlg.get_options()
            if opts['format'] == 'CSV':
                fname, _ = QFileDialog.getSaveFileName(self, 'Export Report', '', 'CSV Files (*.csv);;All Files (*)')
            elif opts['format'] == 'PDF':
                fname, _ = QFileDialog.getSaveFileName(self, 'Export Report', '', 'PDF Files (*.pdf);;All Files (*)')
            else:
                fname, _ = QFileDialog.getSaveFileName(self, 'Export Report', '', 'Excel Files (*.xlsx);;All Files (*)')
            if not fname:
                return
            result = self.api.generate_report(fname, pdf=(opts['format']=='PDF'), excel=(opts['format']=='Excel'))
            self.results_panel.setText(result)
            self.status_bar.showMessage('Report export complete.')
        except Exception as e:
            self.results_panel.setText(f'Report generation error: {e}')
            self.status_bar.showMessage('Report generation error.')
    def staged_construction(self):
        try:
            dlg = StagedConstructionDialog(self)
            if not dlg.exec_():
                return
            opts = dlg.get_options()
            stages = [{} for _ in range(opts['num_stages'])]  # stub: empty stages
            result = self.api.run_staged_construction(stages)
            self.results_panel.setText(f'Staged construction result:\n{result}')
            self.status_bar.showMessage('Staged construction complete.')
        except Exception as e:
            self.results_panel.setText(f'Staged construction error: {e}')
            self.status_bar.showMessage('Staged construction error.')
    def pushover_analysis(self):
        try:
            dlg = PushoverDialog(self, num_nodes=len(self.api.model.nodes) if self.api.model else 1)
            if not dlg.exec_():
                return
            opts = dlg.get_options()
            result = self.api.run_pushover_analysis(opts['control_node'], opts['direction'], opts['steps'], opts['max_disp'])
            self.results_panel.setText('Pushover analysis complete. See plot for capacity curve.')
            self.status_bar.showMessage('Pushover analysis complete.')
            # Plot capacity curve
            disp = result['displacement']
            shear = result['base_shear']
            self.plot_panel.ax.clear()
            self.plot_panel.ax.plot(disp, shear, marker='o')
            self.plot_panel.ax.set_xlabel('Control Node Displacement')
            self.plot_panel.ax.set_ylabel('Base Shear')
            self.plot_panel.ax.set_title('Pushover Capacity Curve')
            self.plot_panel.ax.grid(True)
            self.plot_panel.canvas.draw()
        except Exception as e:
            self.results_panel.setText(f'Pushover analysis error: {e}')
            self.status_bar.showMessage('Pushover analysis error.')
    def time_history_analysis(self):
        try:
            n_dof = len(self.api.model.nodes) * 3 if self.api.model else 1
            dlg = TimeHistoryDialog(self, n_dof=n_dof)
            if not dlg.exec_():
                return
            opts = dlg.get_options()
            dt = opts['dt']
            t_total = opts['t_total']
            dof = opts['dof']
            amp = opts['amp']
            n_steps = int(t_total / dt) + 1
            F_time = np.zeros((n_steps, n_dof))
            F_time[:, dof] = amp  # constant load for demo
            u_hist, v_hist, a_hist = self.api.run_time_history_analysis(F_time, dt, t_total)
            self.results_panel.setText('Time history analysis complete. See plot for displacement time history.')
            self.status_bar.showMessage('Time history analysis complete.')
            # Plot displacement time history for selected DOF
            t = np.linspace(0, t_total, n_steps)
            self.plot_panel.ax.clear()
            self.plot_panel.ax.plot(t, u_hist[:, dof], label=f'DOF {dof}')
            self.plot_panel.ax.set_xlabel('Time (s)')
            self.plot_panel.ax.set_ylabel('Displacement')
            self.plot_panel.ax.set_title('Displacement Time History')
            self.plot_panel.ax.legend()
            self.plot_panel.ax.grid(True)
            self.plot_panel.canvas.draw()
        except Exception as e:
            self.results_panel.setText(f'Time history analysis error: {e}')
            self.status_bar.showMessage('Time history analysis error.')
    def response_spectrum_analysis(self):
        try:
            dlg = ResponseSpectrumDialog(self)
            if not dlg.exec_():
                return
            opts = dlg.get_options()
            # For demo, use a simple spectrum: Sa = 1.0 for all periods
            def spectrum(T):
                return 1.0
            peak_resp = self.api.run_response_spectrum_analysis(spectrum, num_modes=opts['num_modes'])
            self.results_panel.setText('Response spectrum analysis complete. See plot for peak response.')
            self.status_bar.showMessage('Response spectrum analysis complete.')
            # Plot peak response for all DOFs
            dofs = np.arange(len(peak_resp))
            self.plot_panel.ax.clear()
            self.plot_panel.ax.bar(dofs, peak_resp)
            self.plot_panel.ax.set_xlabel('DOF')
            self.plot_panel.ax.set_ylabel('Peak Response')
            self.plot_panel.ax.set_title('Response Spectrum Peak Response')
            self.plot_panel.ax.grid(True)
            self.plot_panel.canvas.draw()
        except Exception as e:
            self.results_panel.setText(f'Response spectrum analysis error: {e}')
            self.status_bar.showMessage('Response spectrum analysis error.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 