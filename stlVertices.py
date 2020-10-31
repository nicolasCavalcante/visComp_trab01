from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import visComp_trab01.Homogeneous as hg
import numpy as np


def open(fName):
    obj = mesh.Mesh.from_file(fName).vectors
    vecStacked = obj.reshape(-1, 3)
    mean = vecStacked.mean(0)
    hull = ConvexHull(vecStacked)
    obj = (obj - mean) / np.power(hull.volume, 1. / 3.)
    affineTransf = hg.rodrigues2Rot(np.array([0, 1, 0]) * - np.pi / 2.)
    obj = affineTransform(obj, affineTransf)
    return obj


def affineTransform(obj, affineTransf, toHomogeneous=False):
    if toHomogeneous:
        shape = obj.shape[0:2]
        onesMat = np.ones(shape + (1,))
        obj = np.concatenate((obj, onesMat), 2)
    obj = obj @ affineTransf.T
    if toHomogeneous:
        obj = obj[..., [0, 1, 2]]
    return obj


def plot(obj, cam, axes3d):
    objStacked = obj.reshape(-1, 3)
    camStacked = cam.reshape(-1, 3)
    objsStacked = np.concatenate((objStacked, camStacked), axis=0)
    pointsMax = objsStacked.max(0)
    pointsMin = objsStacked.min(0)
    pointsMean = (pointsMax + pointsMin) / 2.
    correctAxes = (pointsMax - pointsMin)
    correctAxes = correctAxes.max() / correctAxes
    pointsMin = (pointsMin - pointsMean) * correctAxes + pointsMean
    pointsMax = (pointsMax - pointsMean) * correctAxes + pointsMean

    axes3d.clear()
    axes3d.add_collection3d(mplot3d.art3d.Poly3DCollection(obj,
                                                           facecolor='b',
                                                           edgecolor='k'))
    axes3d.add_collection3d(mplot3d.art3d.Poly3DCollection(cam,
                                                           facecolor='b',
                                                           edgecolor='k'))
    axes3d.set_xlim3d([pointsMin[0], pointsMax[0]])
    axes3d.set_ylim3d([pointsMin[1], pointsMax[1]])
    axes3d.set_zlim3d([pointsMin[2], pointsMax[2]])
    axes3d.set_xlabel('x')
    axes3d.set_ylabel('y')
    axes3d.set_zlabel('z')
    plt.show()
