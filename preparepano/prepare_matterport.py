# Created 2020 by JOANNEUM RESEARCH as part of the ATLANTIS H2020 project
# https://www.joanneum.at
# http://www.atlantis-ar.eu
#
# This tool is part of a project that has received funding from the European 
# Union’s Horizon 2020 research and innovation programme under grant 
# agreement No 951900.


import numpy as np
from PIL import Image
# conversion package for panoramnic images
# https://github.com/sunset1995/py360convert
# can be installed using pip install py360convert
import py360convert 
import zipfile
import os
from os import path
from createpano import *
import sys
import copy




def unzip(basedir,filename):
    print("unpacking "+basedir+"/"+filename)
    with zipfile.ZipFile(basedir+"/"+filename, 'r') as zip_ref:
        zip_ref.extractall(basedir)

def parseCameraParams(filename):
    file = open(filename, 'r') 
  
    paramdict = {}
    
    while True: 
 
        line = file.readline() 
  
        if not line: 
            break
        
        lineparts = line.split(" ")
        if lineparts[0]=="scan":
            (loc,row,ori) = lineparts[1].split("_", 3)
            rowid = int(row[1:])
            (ori,ext) = ori.split(".",2)
            oriid = int(ori)
            
            trmatrix = np.array([[float(lineparts[3]), float(lineparts[4]), float(lineparts[5]), float(lineparts[6])],
                                 [float(lineparts[7]), float(lineparts[8]), float(lineparts[9]), float(lineparts[10])],
                                 [float(lineparts[11]), float(lineparts[12]), float(lineparts[13]), float(lineparts[14])],
                                 [float(lineparts[15]), float(lineparts[16]), float(lineparts[17]), float(lineparts[18])]
                                 ])
            if not(loc in paramdict.keys()):
                paramdict[loc] = {}
            if not(rowid in paramdict[loc].keys()):
                paramdict[loc][rowid] = {}
            paramdict[loc][rowid][oriid] = trmatrix
                
    file.close() 
    
    return paramdict

def correctDepthDistortion(depthImgIn):
    
    depthImg = copy.deepcopy(depthImgIn)
    
    c1 = depthImg.shape[1]/2
    c0 = depthImg.shape[0]/2
    halfFov = default_fov / 2
    
    for i in range(depthImg.shape[1]):
        angle1 = (abs(i-c1)/c1) * halfFov
        for j in range(depthImg.shape[0]):
            d1 = math.tan(angle1) * depthImg[j,i,0]
            angle0 = (abs(j-c0)/c0) * halfFov    
            d0 = math.tan(angle0) * depthImg[j,i,0] 
            diag = math.sqrt(d0*d0 + d1*d1)
            distToCenter = math.sqrt(float(depthImg[j,i,0])*float(depthImg[j,i,0]) + diag*diag)
            corr = distToCenter - depthImg[j,i,0]
            
            if depthImg[j,i,0] >0:
                result = depthImg[j,i,0] + corr
                if result>65535:
                    result = 65535
                depthImg[j,i,0] = result

        
    return depthImg
    

def processFileGroup(basedir,scanId,name,targetDir,extension,isSkyBox,interpolate=True):    
    faceSeq = ['U','B','R','F','L','D']
    
    if not(os.path.exists(targetDir)):
        os.mkdir(targetDir)

    if not(os.path.exists(targetDir+"/"+name)):
        os.mkdir(targetDir+"/"+name)
    
    srcdir = basedir+"/"+scanId+"/"+scanId+"/"+name
    filelist = os.listdir(srcdir)
    filelist.sort()
    filedict = {}
    for filename in filelist:
        if not(filename.endswith(extension)):
            continue
        (namepart,extpart) = filename.split(".", 2)
        tokens = namepart.split("_", 3)
        print("file "+filename)
        srcimg = np.array(Image.open(srcdir+"/"+filename))    
        if srcimg.ndim==2:
            srcimg = np.reshape(srcimg,(srcimg.shape[0],srcimg.shape[1],1))		
        locationId = tokens[0]
        if isSkyBox:
            if not(locationId) in filedict.keys():
                filedict[locationId] = {}
            filedict[locationId][faceSeq[len(filedict[locationId])]] = srcimg
        else:
            if not(locationId) in filedict.keys():
                filedict[locationId] = [ srcimg ]
            else:
                filedict[locationId].append(srcimg)
        
    if not(isSkyBox):
        paramdict = parseCameraParams(basedir+"/"+scanId+"/"+scanId+"/undistorted_camera_parameters/"+scanId+".conf")
            
    for location in filedict.keys():
        if isSkyBox:
            facelist = [ np.fliplr(filedict[location]['F']),
                         filedict[location]['R'],
                         filedict[location]['B'],
                         np.fliplr(filedict[location]['L']),
                         filedict[location]['U'],
                         np.flipud(filedict[location]['D']) ]
            eqrar = py360convert.c2e(facelist, equirectSize[1], equirectSize[0], mode='bilinear', cube_format='list')
            eqrar = np.fliplr(eqrar)
            eqrimg = Image.fromarray(eqrar.astype(np.uint8))
            print("saving "+targetDir+"/"+name+"/"+location+".jpg")
            eqrimg.save(targetDir+"/"+name+"/"+location+".jpg")
        else:
            v = getAngles(paramdict[location])
            blending = True
            if name.startswith("segmentation_maps"):
                blending = False
            isDepth = False
            if name=="undistorted_depth_images":
                isDepth = True
                blending = False
                if warpDepth:
                    for i in range(len(filedict[location])):                       
                        depthimg= correctDepthDistortion(filedict[location][i])           
                        
                        filedict[location][i] = depthimg
                        
            eqrar = combineViews(filedict[location],v,equirectSize,blending,isDepth)
            if name=="undistorted_depth_images":
                array_buffer = eqrar.astype(np.uint16).tobytes()
                eqrimg = Image.new("I", (eqrar.shape[1],eqrar.shape[0]))
                eqrimg.frombytes(array_buffer, 'raw', "I;16")
                #eqrimg = Image.fromarray(eqrar.astype(np.uint16))
                print("saving "+targetDir+"/"+name+"/"+location+".png")
                eqrimg.save(targetDir+"/"+name+"/"+location+".png", "PNG",compress_level=0)
            elif name.startswith("segmentation_maps"):
                eqrimg = Image.fromarray(eqrar.astype(np.uint8))
                print("saving "+targetDir+"/"+name+"/"+location+".png")
                eqrimg.save(targetDir+"/"+name+"/"+location+".png", "PNG",compress_level=0)                
            else:
                eqrimg = Image.fromarray(eqrar.astype(np.uint8))
                print("saving "+targetDir+"/"+name+"/"+location+".jpg")
                eqrimg.save(targetDir+"/"+name+"/"+location+".jpg")
                
        

				

def processScan(matterportHome, matterportEquirect, scanId):			
    if doUnpack:
        unzip(matterportHome+"/"+scanId,"undistorted_camera_parameters.zip")
        unzip(matterportHome+"/"+scanId,"house_segmentations.zip")
        unzip(matterportHome+"/"+scanId,"undistorted_color_images.zip")
        unzip(matterportHome+"/"+scanId,"undistorted_depth_images.zip")
        unzip(matterportHome+"/"+scanId,"matterport_skybox_images.zip")
    
    if not(onlyUnpack):
        #processFileGroup(matterportHome,scanId,"matterport_skybox_images",matterportEquirect+"/"+scanId,"jpg",True)
        #processFileGroup(matterportHome,scanId,"undistorted_color_images",matterportEquirect+"/"+scanId,"jpg",False)
        processFileGroup(matterportHome,scanId,"undistorted_depth_images",matterportEquirect+"/"+scanId,"png",False)
        #processFileGroup(matterportHome,scanId,"segmentation_maps_classes",matterportEquirect+"/"+scanId,"png",False,False)
        #processFileGroup(matterportHome,scanId,"segmentation_maps_instances",matterportEquirect+"/"+scanId,"png",False,False)

# MAIN
matterportHome = "Y:/Datasets/Matterport/v1/scans"
matterportEquirect = "Y:/Datasets/Matterport/v1/equirect"
doUnpack = False
onlyUnpack = False
equirectSize = [1920,1080]

warpDepth = True

processScan(matterportHome, matterportEquirect,"1LXtFkjw3qL")

#filelist = os.listdir(matterportHome)
#for filename in filelist:
#    processScan(matterportHome, matterportEquirect,filename)
    
# TEST houses
testlist = [
"2t7WUuJeko7",
"5ZKStnWn8Zo",
"ARNzJeq3xxb",
"fzynW3qQPVF",
"jtcxE69GiFV",
"pa4otMbVnkk",
"q9vSo1VnCiC",
"rqfALeAoiTq",
"UwV83HsGsw3",
"wc2JMjhGNzB",
"WYY7iVyf5p8",
"YFuZgdQ5vWj",
"yqstnuAEVhm",
"YVUC4YcDtcY",
"gxdoqLR6rwA",
"gYvKGZ5eRqb",
"RPmz2sHmrrY",
"Vt2qJdWjCF2"
]

#for filename in testlist:
#    processScan(matterportHome, matterportEquirect,filename)