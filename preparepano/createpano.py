# Created 2020 by JOANNEUM RESEARCH as part of the ATLANTIS H2020 project
# https://www.joanneum.at
# http://www.atlantis-ar.eu
#
# This tool is part of a project that has received funding from the European 
# Union’s Horizon 2020 research and innovation programme under grant 
# agreement No 951900.

# ported and extended code from https://github.com/yindaz/PanoBasic

import typing
import numpy as np
from numpy.linalg import inv
import math
import scipy
from packaging import version
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import *
import cv2
from PIL import Image

# definitions following PanoBasic
refview = (1,3)
imcutout = [[0,1013],[0,1254]]
default_fov = 1.06

# adjustment to match Matterport Skybox
xoffset = math.pi / 3.0  # 60 degs

# get hor/vert angles for each view, starting from Matterport matrices (inverse of extrinsic)
def get_angles(matrixDict) -> np.array:
    v = np.zeros((18,2))
    rot_ctor = Rot.from_matrix\
        if version.parse(scipy.__version__) >= version.parse('1.4.0')\
        else Rot.from_dcm
    euler = np.array([ 
        rot_ctor(inv(np.transpose(matrixDict[0][0]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[0][1]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[0][2]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[0][3]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[0][4]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[0][5]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[1][0]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[1][1]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[1][2]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[1][3]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[1][4]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[1][5]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[2][0]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[2][1]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[2][2]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[2][3]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[2][4]))[0:3,0:3]).as_euler('xyz'),
        rot_ctor(inv(np.transpose(matrixDict[2][5]))[0:3,0:3]).as_euler('xyz') 
    ])
    step = 6
    for i in range(3):
        v[i * step:(i+1) * step, 0] = np.array([
            euler[refview[0]*step+refview[1],2] - euler[i*step+0,2],
            euler[refview[0]*step+refview[1],2] - euler[i*step+1,2],
            euler[refview[0]*step+refview[1],2] - euler[i*step+2,2],
            euler[refview[0]*step+refview[1],2] - euler[i*step+3,2],
            euler[refview[0]*step+refview[1],2] - euler[i*step+4,2],
            euler[refview[0]*step+refview[1],2] - euler[i*step+5,2] 
        ])
        v[i * step: (i+1) * step, 1] = np.array([
            euler[refview[0]*step+refview[1],0] - euler[i*step+0,0],
            euler[refview[0]*step+refview[1],0] - euler[i*step+1,0],
            euler[refview[0]*step+refview[1],0] - euler[i*step+2,0],
            euler[refview[0]*step+refview[1],0] - euler[i*step+3,0],
            euler[refview[0]*step+refview[1],0] - euler[i*step+4,0],
            euler[refview[0]*step+refview[1],0] - euler[i*step+5,0] 
        ])
    v[:, 0] = v[:, 0] + xoffset
    v[v > math.pi] -= 2 * math.pi        
    v[:, 1] = -v[:, 1]
    return v

# set blending false for label maps
def combine_views(
    images: typing.List[np.array],
    v: np.array,
    outsize: typing.Tuple[int, int],
    blending: bool=True,
    depth: bool=False
):
    nchannels = images[0].shape[2]
    pano = np.zeros((outsize[1],outsize[0],nchannels))
    pano_w = np.zeros((outsize[1],outsize[0],nchannels))
    for i in range(len(images)):
        if images[i].size < 3:
            continue
        sphere_img, validMap = im2sphere(
            images[i][imcutout[0][0]:imcutout[0][1],imcutout[1][0]:imcutout[1][1]],
            default_fov, 
            outsize[0], 
            outsize[1], 
            v[i,0], 
            v[i,1], 
            blending,
            i,
            depth
        )         
        sphere_img[validMap<0.00000001] = 0
        if blending:
            pano = pano + sphere_img
        else:
            if depth:
                sphere_img[:,:,0] = sphere_img[:,:,0] * validMap
                pano = pano + sphere_img 

            else:
                pano[sphereImg>0] = sphereImg[sphereImg>0] 
        pano_w[:,:,0] = pano_w[:,:,0] + validMap
        if nchannels>1:
            pano_w[:,:,1] = pano_w[:,:,1] + validMap
            pano_w[:,:,2] = pano_w[:,:,2] + validMap
    pano[pano_w==0] = 0
    pano_w[pano_w==0] = 1
    if blending or depth:
        pano = np.divide(pano, pano_w)
    return pano

def im2sphere(
    im: np.array,
    imHoriFOV: float,
    sphereW: int,
    sphereH: int,
    x: float,
    y: float,
    interpolate: bool,
    nr: int,
    weightByCenterDist: bool = False
):
    # map pixel in panorama to viewing direction
    TX, TY = np.meshgrid(np.array(range(sphereW)), np.array(range(sphereH)))
    TX = TX.flatten('F')
    TY = TY.flatten('F')
    ANGx = ((TX - (sphereW / 2) - 0.5) / sphereW) * math.pi * 2.0
    ANGy = (-(TY - (sphereH / 2) - 0.5) / sphereH) * math.pi
    # compute the radius of ball
    imH = im.shape[0]
    imW = im.shape[1]
    R = (imW/2) / math.tan(imHoriFOV/2)
    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * math.cos(y) * math.sin(x)
    y0 = R * math.cos(y) * math.cos(x)
    z0 = R * math.sin(y)
    # plane function: x0(x-x0)+y0(y-y0)+z0(z-z0)=0
    # view line: x/alpha=y/belta=z/gamma
    # alpha=cos(phi)sin(theta);  belta=cos(phi)cos(theta);  gamma=sin(phi)
    alpha = np.multiply(np.cos(ANGy), np.sin(ANGx))
    beta = np.multiply(np.cos(ANGy), np.cos(ANGx))
    gamma = np.sin(ANGy)
    # solve for intersection of plane and viewing line: [x1 y1 z1]
    division = x0 * alpha + y0 * beta + z0 * gamma
    x1 = R * R * np.divide(alpha, division)
    y1 = R * R * np.divide(beta, division)
    z1 = R * R * np.divide(gamma, division)
    # vector in plane: [x1-x0 y1-y0 z1-z0]
    # positive x vector: vecposX = [cos(x) -sin(x) 0]
    # positive y vector: vecposY = [x0 y0 z0] x vecposX
    vec = np.transpose(np.array([x1 - x0, y1 - y0, z1 - z0]))
    vecposX = np.transpose(np.array([math.cos(x), -math.sin(x), 0]))
    deltaX = np.dot(vecposX,np.transpose(vec)) / np.sqrt(np.dot(vecposX,np.transpose(vecposX)))
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    deltaY = np.dot(vecposY,np.transpose(vec)) / np.sqrt(np.dot(vecposY,np.transpose(vecposY)))
    # convert to im coordinates
    Px = np.reshape(deltaX, (sphereH, sphereW),'F') + (imW+1)/2
    Py = np.reshape(deltaY, (sphereH, sphereW),'F') + (imH+1)/2
    # warp image
    sphere_img = warp_image_fast(im, Px, Py, interpolate, (sphereW, sphereH), nr)
    validMap = np.zeros((sphere_img.shape[0], sphere_img.shape[1]))
    validMap[:,:] = np.logical_not(np.isnan(sphere_img[:,:,0])).astype(float)

    if weightByCenterDist:
        weightIm = np.zeros((im.shape[0],im.shape[1],1))
        c0 = im.shape[0] / 2
        c1 = im.shape[1] / 2
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                weightIm[i,j,0] = (1 - abs(c0 - i)/c0) * (1 - abs(c1 - j)/c1)
                
        weightImWarped = warp_image_fast(weightIm, Px, Py,False,(sphereW,sphereH),nr)

        validMap = weightImWarped[:,:,0]
        validMap[sphere_img[:,:,0]<1] = 0

    else:
        validMap[sphere_img[:,:,0]<0] = 0
    # view direction: [alpha belta gamma]
    # contacting point direction: [x0 y0 z0]
    # so division>0 are valid region
    validMap[np.reshape(division, (sphereH, sphereW), 'F') < 0] = 0
    return sphere_img, validMap
   
def warp_image_fast(
    im: np.array,
    XXdense: np.array,
    YYdense: np.array,
    interpolate: bool,
    outsize: typing.Tuple[int, int],
    nr: int
):
    nchannels = im.shape[2]
    minX = max(1,math.floor(XXdense.min()) - 1)
    minY = max(1,math.floor(YYdense.min()) - 1)
    maxX = min(im.shape[1], math.ceil(XXdense.max()) + 1)
    maxY = min(im.shape[0], math.ceil(YYdense.max()) + 1)
    im = im[minY:maxY, minX:maxX, :]
    im_warp = np.zeros((outsize[1], outsize[0], nchannels))
    intermode = cv2.INTER_NEAREST
    if interpolate:
        intermode = cv2.INTER_LINEAR
    for i in range(nchannels):
        mapx = XXdense-minX + 1
        mapy = YYdense-minY + 1
        im_warp[:,:,i] = cv2.remap(im[:,:,i].astype(np.float32), mapx.astype(np.float32), mapy.astype(np.float32), 
               interpolation=intermode, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1,-1,-1) )
    return im_warp




