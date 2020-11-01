from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import visComp_trab01.Homogeneous as hg
import numpy as np
import numpy.matlib


def open(fName):
    vectors = mesh.Mesh.from_file(fName).vectors
    cartP = vectors.reshape(-1, 3).T
    mean = cartP.mean(1)
    hull = ConvexHull(cartP.T)
    cartP = (cartP - np.expand_dims(mean, 1)) / np.power(hull.volume, 1. / 3.)
    correctOrientation = hg.mat3D((0, 0, 0, 0, - np.pi / 2., 0))
    correctOrientation = hg.mat3D((0, 0, 0.2, 0,
                                   0, np.pi / 2.)) @ correctOrientation
    homoP = hg.cart2homo(cartP)
    return correctOrientation @ homoP


def plot(obj, cam, camTransformE, camTransformI, axes2d, axes3d):
    def toVertices(homoP):
        cartP = hg.homo2cart(homoP)
        return cartP.T.reshape(-1, 3, 3)

    def setLim3d(obj, cam, axes3d):
        objStacked = hg.homo2cart(obj).T
        camStacked = hg.homo2cart(cam).T
        objsStacked = np.concatenate((objStacked, camStacked), axis=0)
        pointsMax = objsStacked.max(0)
        pointsMin = objsStacked.min(0)
        pointsMean = (pointsMax + pointsMin) / 2.
        correctAxes = (pointsMax - pointsMin)
        correctAxes = correctAxes.max() / correctAxes
        pointsMin = (pointsMin - pointsMean) * correctAxes + pointsMean
        pointsMax = (pointsMax - pointsMean) * correctAxes + pointsMean
        axes3d.set_xlim3d([pointsMin[0], pointsMax[0]])
        axes3d.set_ylim3d([pointsMin[1], pointsMax[1]])
        axes3d.set_zlim3d([pointsMin[2], pointsMax[2]])

    def addQuiver(ax, origin, length, color):
        length = length - origin
        ax.quiver(origin[0], origin[1], origin[2],
                  length[0], length[1], length[2], color=color)
    # 3D plot
    axes3d.clear()
    axes3d.add_collection3d(mplot3d.art3d.Poly3DCollection(toVertices(obj),
                                                           facecolor='b',
                                                           edgecolor='k'))
    addQuiver(axes3d, cam[:, 0], cam[:, 1], "red")
    addQuiver(axes3d, cam[:, 0], cam[:, 2], "green")
    addQuiver(axes3d, cam[:, 0], cam[:, 3], "blue")

    axes3d.set_xlabel('x')
    axes3d.set_ylabel('y')
    axes3d.set_zlabel('z')
    setLim3d(obj, cam, axes3d)
    # 2D plot
    objFromCam = hg.mat3Dinv(camTransformE) @ obj
    objFromCamV = toVertices(objFromCam)
    removeNegativeFromImg = np.min(objFromCamV[:, :, 2], axis=1)
    objFromCamV = objFromCamV[removeNegativeFromImg > 0, ...]
    closestPlotLast = np.max(objFromCamV[:, :, 2], axis=1).argsort()
    objFromCamV = objFromCamV[closestPlotLast[::-1], :, :]
    objFromCam = hg.cart2homo(objFromCamV.reshape(-1, 3).T)
    camImg = camTransformI @ objFromCam
    camImg = np.true_divide(camImg, camImg[2, :])
    camImgVectors = camImg.T.reshape(-1, 3, 3)
    for x in np.arange(camImgVectors.shape[0]):
        axes2d.add_patch(plt.Polygon(camImgVectors[x, :, :-1],
                                     facecolor='b', edgecolor='k'))
    axes2d.set_xlim(0, 1920)
    axes2d.set_ylim(0, 1080)
    axes2d.set_xlabel('x')
    axes2d.set_ylabel('y')
    axes2d.xaxis.tick_top()
    axes2d.xaxis.set_label_position('top')
    axes2d.invert_yaxis()
    axes2d.set_aspect('equal')
    plt.show()
