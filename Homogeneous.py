# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:29:09 2020

@author: Nicolas
"""
import numpy as np
from scipy.linalg import block_diag


def cart2homo(cartVector):
    """

    Parameters
    ----------
    cartVector : TYPE
        DESCRIPTION.

    Returns
    -------
    homoVector : TYPE
        DESCRIPTION.

    """
    cartVector = np.asarray(cartVector, dtype=np.float32)
    homoVector = np.vstack((cartVector, np.ones(cartVector.shape[1])))
    return homoVector


def homo2cart(homoVector):
    homoVector = np.asarray(homoVector, dtype=np.float32)
    cartVector = homoVector[0:3, :]
    return cartVector


def skew(cartVector):
    """
    This function returns a numpy array with the skew symmetric cross product
    matrix for vector. The skew symmetric cross product matrix is defined such
    that np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross
    product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -cartVector[2], cartVector[1]],
                     [cartVector[2], 0, -cartVector[0]],
                     [-cartVector[1], cartVector[0], 0]])


def rotMat2D(th):
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th), np.cos(th)]], dtype=np.float32)


def mat2D(transformVector):
    tx, ty, rz = transformVector
    T = np.array([tx, ty], dtype=np.float32)
    R = rotMat2D(rz)
    return np.r_[np.c_[R, T], [[0., 0., 1.]]]


def mat2Dseq(transformVectorseq):
    H2 = np.eye(3)
    for i in range(transformVectorseq.shape[1]):
        H2 = H2 @ mat2D(transformVectorseq[:, i])
    return H2


def mat2Dinv(H2):
    R = H2[0:2, 0:2]
    T = H2[0:2, 2]
    return np.r_[np.c_[R.T, -R.T @ T], [[0., 0., 1.]]]


def rotMat3D(th):
    thx, thy, thz = th
    Rx, Ry, Rz = rotMat2D(th).swapaxes(0, 2).swapaxes(1, 2)
    Rx = block_diag(1, Rx)
    Ry = np.reshape(Ry, Ry.size)
    Ry = np.array([[Ry[0], 0., Ry[2]], [0., 1., 0.], [Ry[1], 0., Ry[3]]])
    Rz = block_diag(Rz, 1)
    return Rx @ Ry @ Rz


def mat3D(transformVector):
    tx, ty, tz, rx, ry, rz = transformVector
    T = np.array([tx, ty, tz], dtype=np.float32)
    R = rotMat3D([rx, ry, rz])
    return np.r_[np.c_[R, T], [[0, 0, 0, 1]]]


def mat3Dseq(transformVectorseq):
    H3 = np.eye(4)
    for i in range(transformVectorseq.shape[1]):
        H3 = H3 @ mat3D(transformVectorseq[:, i])
    return H3


def mat3Dinv(H3):
    R = H3[0:3, 0:3]
    T = H3[0:3, 3]
    return np.r_[np.c_[R.T, -R.T @ T], [[0., 0., 0., 1.]]]


def rodrigues2Rot(w):
    mod = np.linalg.norm(w)
    if mod == 0.0:
        return np.eye(3)
    skn = skew(w / mod)
    return np.eye(3) + skn * np.sin(mod) + skn @ skn * (1. - np.cos(mod))


def rot2Rogrigues(R):
    mod = np.arccos((np.trace(R) - 1) / 2)
    return mod / 2 / np.sin(mod) * np.array([R[2, 1] - R[1, 2],
                                             R[0, 2] - R[2, 0],
                                             R[1, 0] - R[0, 1]])


if __name__ == '__main__':
    theta = 30 / 180 * np.pi
    R2 = rotMat2D(theta)
    homoVector = cart2homo([1, 2, 3])
    cartVector = homo2cart(homoVector)
    transformVector = np.array([1., 1, theta])
    H2 = mat2D(transformVector)
    transformVectorseq = np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]])
    H2seq = mat2Dseq(transformVectorseq)
    H2inv = mat2Dinv(H2)

    theta3 = np.array([1, 2, 3.])
    R3 = rotMat3D(theta3)
    transformVector3 = np.array([1, 2, 3, 4, 5, 6])
    H3 = mat3D(transformVector3)
    transformVectorseq3 = np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.],
                                    [0., 0., 1.],
                                    [0., 0., 0.],
                                    [0., 0., 0.]])
    H3seq = mat3Dseq(transformVectorseq3)
    H3inv = mat3Dinv(H3)
    print(skew(theta3))
    R = rodrigues2Rot([0, 0., np.pi / 2])
    print(R)
    print(rot2Rogrigues(R))
