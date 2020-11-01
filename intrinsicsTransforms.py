import numpy as np


def to2dPinhole(f):
    cannonicalProjMat = np.array([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.]])
    return np.diag([f, f, 1.]) @ cannonicalProjMat


def img2sensorMat(sx, sy, ox, oy):
    return np.array([[sx, 0, ox],
                     [0, sy, oy],
                     [0, 0, 1.]])
