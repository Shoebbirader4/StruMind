import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Example model data (replace with real model import)
beams = [((0, 0, 0), (2, 0, 0)), ((2, 0, 0), (2, 2, 0))]
shells = [[(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]]
solids = [[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]

angle_x = 20
angle_y = 30
zoom = -8
pan_x = 0
pan_y = 0
mouse_last = [0, 0]
mouse_btn = None

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(pan_x, pan_y, zoom)
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)
    # Draw beams as lines
    glColor3f(1, 0, 0)
    glLineWidth(3)
    glBegin(GL_LINES)
    for start, end in beams:
        glVertex3fv(start)
        glVertex3fv(end)
    glEnd()
    # Draw shells as quads
    glColor3f(0, 1, 0)
    for quad in shells:
        glBegin(GL_QUADS)
        for v in quad:
            glVertex3fv(v)
        glEnd()
    # Draw solids as wireframe cubes
    glColor3f(0, 0, 1)
    for cube in solids:
        edges = [
            (0,1),(1,2),(2,3),(3,0), # bottom
            (4,5),(5,6),(6,7),(7,4), # top
            (0,4),(1,5),(2,6),(3,7)  # sides
        ]
        glBegin(GL_LINES)
        for i,j in edges:
            glVertex3fv(cube[i])
            glVertex3fv(cube[j])
        glEnd()
    glutSwapBuffers()

def mouse(button, state, x, y):
    global mouse_btn, mouse_last
    if state == GLUT_DOWN:
        mouse_btn = button
        mouse_last = [x, y]
    else:
        mouse_btn = None

def motion(x, y):
    global angle_x, angle_y, pan_x, pan_y, mouse_last
    dx = x - mouse_last[0]
    dy = y - mouse_last[1]
    if mouse_btn == GLUT_LEFT_BUTTON:
        angle_x += dy
        angle_y += dx
    elif mouse_btn == GLUT_RIGHT_BUTTON:
        pan_x += dx * 0.01
        pan_y -= dy * 0.01
    mouse_last[:] = [x, y]
    glutPostRedisplay()

def mouse_wheel(button, dir, x, y):
    global zoom
    if dir > 0:
        zoom += 0.5
    else:
        zoom -= 0.5
    glutPostRedisplay()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w/float(h or 1), 0.1, 100)
    glMatrixMode(GL_MODELVIEW)

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b'Structural Engine OpenGL Viewer')
    glEnable(GL_DEPTH_TEST)
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    try:
        glutMouseWheelFunc(mouse_wheel)
    except:
        pass  # Not all GLUTs support mouse wheel
    glutMainLoop()

if __name__ == '__main__':
    main() 