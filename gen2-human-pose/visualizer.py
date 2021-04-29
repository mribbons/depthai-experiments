import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
import math
import numpy as np
from OpenGL.GL import glLineWidth, glBegin, GL_LINES, glColor3f, glVertex3fv, glEnd, glPointSize, GL_POINTS, glVertex3f, \
    glScaled, GLfloat, glGetFloatv, GL_MODELVIEW_MATRIX, glRotatef, glTranslatef, glClear, GL_COLOR_BUFFER_BIT, \
    GL_DEPTH_BUFFER_BIT
from OpenGL.GLU import gluPerspective

lastPosX = 0
lastPosY = 0
zoomScale = 1.0
dataL = 0
xRot = 0
yRot = 0
zRot = 0


def draw(points, lines=[]):
    glLineWidth(1.5)

    glBegin(GL_LINES)    
    for line in lines:
        glColor3f(line[6]/255.0, line[7]/255.0, line[8]/255.0)
        glVertex3fv([line[0], line[1], line[2]])
        glVertex3fv([line[3], line[4], line[5]])
    glEnd()

    glPointSize(3.0)
    glBegin(GL_POINTS)
    for point in points:
        glColor3f(point[3]/255.0, point[4]/255.0, point[5]/255.0)
        glVertex3f(point[0], point[1], point[2])
    glEnd()

def mouseMove(event):
    global lastPosX, lastPosY, zoomScale, xRot, yRot, zRot

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
        glScaled(1.05, 1.05, 1.05)
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
        glScaled(0.95, 0.95, 0.95)

    if event.type == pygame.MOUSEMOTION:
        x, y = event.pos
        dx = x - lastPosX
        dy = y - lastPosY

        mouseState = pygame.mouse.get_pressed()
        if mouseState[0]:
            modelView = (GLfloat * 16)()
            mvm = glGetFloatv(GL_MODELVIEW_MATRIX, modelView)

            temp = (GLfloat * 3)()
            temp[0] = modelView[0] * dy + modelView[1] * dx
            temp[1] = modelView[4] * dy + modelView[5] * dx
            temp[2] = modelView[8] * dy + modelView[9] * dx
            norm_xy = math.sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2])
            glRotatef(math.sqrt(dx * dx + dy * dy), temp[0] / norm_xy, temp[1] / norm_xy, temp[2] / norm_xy)

        lastPosX = x
        lastPosY = y


def initialize_OpenGL():
    pygame.init()

    display = (300, 300)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL, RESIZABLE)

    gluPerspective(45, (1.0 * display[0] / display[1]), 0.1, 50.0)
    glTranslatef(-0.5, 0, -0.5)


r = 0.05
def start_OpenGL(points, lines):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        mouseMove(event)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw(points, lines)
    pygame.display.flip()
    pygame.time.wait(1)


def get_vector_direction(camera_position, landmark):
    vector = []

    for i in range(3):
        vector.append(landmark[i] - camera_position[i])

    return np.array(vector)


def get_vector_intersection(left_vector, left_camera_position, right_vector, right_camera_position):
    n = np.cross(left_vector, right_vector)
    n1 = np.cross(left_vector, n)
    n2 = np.cross(right_vector, n)

    top = np.dot(np.subtract(right_camera_position, left_camera_position), n2)
    bottom = np.dot(left_vector, n2)
    divided = top / bottom
    mult = divided * left_vector
    c1 = left_camera_position + mult

    top = np.dot(np.subtract(left_camera_position, right_camera_position), n1)
    bottom = np.dot(right_vector, n1)
    divided = top / bottom
    mult = divided * right_vector
    c2 = right_camera_position + mult

    center = (c1 + c2) / 2
    return center
