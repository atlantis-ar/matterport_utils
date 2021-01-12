#!/usr/bin/env python3

# Created 2020 by JOANNEUM RESEARCH as part of the ATLANTIS H2020 project
# https://www.joanneum.at
# http://www.atlantis-ar.eu
#
# This tool is part of a project that has received funding from the European 
# Union’s Horizon 2020 research and innovation programme under grant 
# agreement No 951900.

import datetime
import json
import os
import re
import fnmatch
import numpy as np
from  pycococreatortools import pycococreatortools
from PIL import Image
import csv

from skimage.morphology import erosion, dilation, opening, closing, white_tophat, disk
from skimage import measure
from scipy.ndimage import binary_fill_holes
from copy import copy

import sys, argparse
# params
parser = argparse.ArgumentParser()
parser.add_argument('--matterport_root_dir', required=True, help='input path to root directory (up to v1)')
parser.add_argument('--matterport_scene_dir', required=True, help='input path to house (with color, depth and instance_filt) to convert')
parser.add_argument('--matterport_annotation_dir', required=True, help='input path to the annotation files (json)')
parser.add_argument('--matterport_house_id', required=True, nargs='+', default=['2t7WUuJeko7'],
		help='Specify list of houses to be processed')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--coco_annotation_dir', required=True, help='output root for annotations')
parser.add_argument('--coco_annotation_file', required=True, help='filename for annotation json')
parser.add_argument('--tolerance', type=int, default=2, help='mask smoothing, higher is smoother')
parser.add_argument('--class_labels',default="nyu40",help='type of class labels to output: nyu40, coco (will only include overlapping classes)')
parser.add_argument('--discard_wrap_around_regions',type=int,default=0,help='if >0, remove regions not entirely included in the center part of the specified width')
parser.add_argument('--clean_masks',dest='clean_masks',action='store_true',help='perform morphological operations and hole filling on masks')
parser.set_defaults(export_depth_images=False, export_color_images=False)
parser.add_argument('--min_region_area',type=float,default=0,help='requrie min size of region to be kept the given fraction of the image area')
parser.add_argument('--do_stats',dest='do_stats',action='store_true',help='calculate statistics on object sizes and recurrent objects in other views')
opt = parser.parse_args()
print(opt)


ROOT_DIR = opt.matterport_root_dir # 'some/path/matterport/v1'
SCENE_DIR = opt.matterport_scene_dir # 'equirect/<house>'
HOUSES = opt.matterport_house_id
SRC_ANNOTATION_DIR = opt.matterport_annotation_dir #  'ply/<house>/sphere_points_smooth'
ANNOTATION_DIR = opt.coco_annotation_dir # os.path.join(ROOT_DIR, "annotations")
ANNOTATION_FILE = opt.coco_annotation_file
TOLERANCE = opt.tolerance
CLASS_LABELS = opt.class_labels

OUTPUT = os.path.join(ANNOTATION_DIR, ANNOTATION_FILE)


# nyu40id copied from ScanNet site
NYU40_CATEGORIES = [
{"supercategory": "shape", "id": 1, "name": "wall" },
{"supercategory": "shape", "id": 2, "name": "floor" },
{"supercategory": "shape", "id": 3, "name": "cabinet" },
{"supercategory": "shape", "id": 4, "name": "bed" },
{"supercategory": "shape", "id": 5, "name": "chair" },
{"supercategory": "shape", "id": 6, "name": "sofa" },
{"supercategory": "shape", "id": 7, "name": "table" },
{"supercategory": "shape", "id": 8, "name": "door" },
{"supercategory": "shape", "id": 9, "name": "window" },
{"supercategory": "shape", "id": 10, "name": "bookshelf" },
{"supercategory": "shape", "id": 11, "name": "picture" },
{"supercategory": "shape", "id": 12, "name": "counter" },
{"supercategory": "shape", "id": 13, "name": "blinds" },
{"supercategory": "shape", "id": 14, "name": "desk" },
{"supercategory": "shape", "id": 15, "name": "shelves" },
{"supercategory": "shape", "id": 16, "name": "curtain" },
{"supercategory": "shape", "id": 17, "name": "dresser" },
{"supercategory": "shape", "id": 18, "name": "pillow" },
{"supercategory": "shape", "id": 19, "name": "mirror" },
{"supercategory": "shape", "id": 20, "name": "floor mat" },
{"supercategory": "shape", "id": 21, "name": "clothes" },
{"supercategory": "shape", "id": 22, "name": "ceiling" },
{"supercategory": "shape", "id": 23, "name": "books" },
{"supercategory": "shape", "id": 24, "name": "refridgerator" },
{"supercategory": "shape", "id": 25, "name": "television" },
{"supercategory": "shape", "id": 26, "name": "paper" },
{"supercategory": "shape", "id": 27, "name": "towel" },
{"supercategory": "shape", "id": 28, "name": "shower curtain" },
{"supercategory": "shape", "id": 29, "name": "box" },
{"supercategory": "shape", "id": 30, "name": "whiteboard" },
{"supercategory": "shape", "id": 31, "name": "person" },
{"supercategory": "shape", "id": 32, "name": "nightstand" },
{"supercategory": "shape", "id": 33, "name": "toilet" },
{"supercategory": "shape", "id": 34, "name": "sink" },
{"supercategory": "shape", "id": 35, "name": "lamp" },
{"supercategory": "shape", "id": 36, "name": "bathtub" },
{"supercategory": "shape", "id": 37, "name": "bag" },
{"supercategory": "shape", "id": 38, "name": "otherstructure" },
{"supercategory": "shape", "id": 39, "name": "otherfurniture" },
{"supercategory": "shape", "id": 40, "name": "otherprop" }
]


# COCO categories
# TRUE IDs - not used here
# COCO_CATEGORIES = [
# {"supercategory": "shape", "id": 1, "name": "person"},
# {"supercategory": "shape", "id": 2, "name": "bicycle"},
# {"supercategory": "shape", "id": 3, "name": "car"},
# {"supercategory": "shape", "id": 4, "name": "motorcycle"},
# {"supercategory": "shape", "id": 5, "name": "airplane"},
# {"supercategory": "shape", "id": 6, "name": "bus"},
# {"supercategory": "shape", "id": 7, "name": "train"},
# {"supercategory": "shape", "id": 8, "name": "truck"},
# {"supercategory": "shape", "id": 9, "name": "boat"},
# {"supercategory": "shape", "id": 10, "name": "traffic light"},
# {"supercategory": "shape", "id": 11, "name": "fire hydrant"},
# {"supercategory": "shape", "id": 13, "name": "stop sign"},
# {"supercategory": "shape", "id": 14, "name": "parking meter"},
# {"supercategory": "shape", "id": 15, "name": "bench"},
# {"supercategory": "shape", "id": 16, "name": "bird"},
# {"supercategory": "shape", "id": 17, "name": "cat"},
# {"supercategory": "shape", "id": 18, "name": "dog"},
# {"supercategory": "shape", "id": 19, "name": "horse"},
# {"supercategory": "shape", "id": 20, "name": "sheep"},
# {"supercategory": "shape", "id": 21, "name": "cow"},
# {"supercategory": "shape", "id": 22, "name": "elephant"},
# {"supercategory": "shape", "id": 23, "name": "bear"},
# {"supercategory": "shape", "id": 24, "name": "zebra"},
# {"supercategory": "shape", "id": 25, "name": "giraffe"},
# {"supercategory": "shape", "id": 27, "name": "backpack"},
# {"supercategory": "shape", "id": 28, "name": "umbrella"},
# {"supercategory": "shape", "id": 31, "name": "handbag"},
# {"supercategory": "shape", "id": 32, "name": "tie"},
# {"supercategory": "shape", "id": 33, "name": "suitcase"},
# {"supercategory": "shape", "id": 34, "name": "frisbee"},
# {"supercategory": "shape", "id": 35, "name": "skis"},
# {"supercategory": "shape", "id": 36, "name": "snowboard"},
# {"supercategory": "shape", "id": 37, "name": "sports ball"},
# {"supercategory": "shape", "id": 38, "name": "kite"},
# {"supercategory": "shape", "id": 39, "name": "baseball bat"},
# {"supercategory": "shape", "id": 40, "name": "baseball glove"},
# {"supercategory": "shape", "id": 41, "name": "skateboard"},
# {"supercategory": "shape", "id": 42, "name": "surfboard"},
# {"supercategory": "shape", "id": 43, "name": "tennis racket"},
# {"supercategory": "shape", "id": 44, "name": "bottle"},
# {"supercategory": "shape", "id": 46, "name": "wine glass"},
# {"supercategory": "shape", "id": 47, "name": "cup"},
# {"supercategory": "shape", "id": 48, "name": "fork"},
# {"supercategory": "shape", "id": 49, "name": "knife"},
# {"supercategory": "shape", "id": 50, "name": "spoon"},
# {"supercategory": "shape", "id": 51, "name": "bowl"},
# {"supercategory": "shape", "id": 52, "name": "banana"},
# {"supercategory": "shape", "id": 53, "name": "apple"},
# {"supercategory": "shape", "id": 54, "name": "sandwich"},
# {"supercategory": "shape", "id": 55, "name": "orange"},
# {"supercategory": "shape", "id": 56, "name": "broccoli"},
# {"supercategory": "shape", "id": 57, "name": "carrot"},
# {"supercategory": "shape", "id": 58, "name": "hot dog"},
# {"supercategory": "shape", "id": 59, "name": "pizza"},
# {"supercategory": "shape", "id": 60, "name": "donut"},
# {"supercategory": "shape", "id": 61, "name": "cake"},
# {"supercategory": "shape", "id": 62, "name": "chair"},
# {"supercategory": "shape", "id": 63, "name": "couch"},
# {"supercategory": "shape", "id": 64, "name": "potted plant"},
# {"supercategory": "shape", "id": 65, "name": "bed"},
# {"supercategory": "shape", "id": 67, "name": "dining table"},
# {"supercategory": "shape", "id": 70, "name": "toilet"},
# {"supercategory": "shape", "id": 72, "name": "tv"},
# {"supercategory": "shape", "id": 73, "name": "laptop"},
# {"supercategory": "shape", "id": 74, "name": "mouse"},
# {"supercategory": "shape", "id": 75, "name": "remote"},
# {"supercategory": "shape", "id": 76, "name": "keyboard"},
# {"supercategory": "shape", "id": 77, "name": "cell phone"},
# {"supercategory": "shape", "id": 78, "name": "microwave"},
# {"supercategory": "shape", "id": 79, "name": "oven"},
# {"supercategory": "shape", "id": 80, "name": "toaster"},
# {"supercategory": "shape", "id": 81, "name": "sink"},
# {"supercategory": "shape", "id": 82, "name": "refrigerator"},
# {"supercategory": "shape", "id": 84, "name": "book"},
# {"supercategory": "shape", "id": 85, "name": "clock"},
# {"supercategory": "shape", "id": 86, "name": "vase"},
# {"supercategory": "shape", "id": 87, "name": "scissors"},
# {"supercategory": "shape", "id": 88, "name": "teddy bear"},
# {"supercategory": "shape", "id": 89, "name": "hair drier"},
# {"supercategory": "shape", "id": 90, "name": "toothbrush"}
# ]

# ids with linear numbering
COCO_CATEGORIES = [
{"supercategory": "shape", "id": 1, "name": "person"},
{"supercategory": "shape", "id": 2, "name": "bicycle"},
{"supercategory": "shape", "id": 3, "name": "car"},
{"supercategory": "shape", "id": 4, "name": "motorcycle"},
{"supercategory": "shape", "id": 5, "name": "airplane"},
{"supercategory": "shape", "id": 6, "name": "bus"},
{"supercategory": "shape", "id": 7, "name": "train"},
{"supercategory": "shape", "id": 8, "name": "truck"},
{"supercategory": "shape", "id": 9, "name": "boat"},
{"supercategory": "shape", "id": 10, "name": "traffic light"},
{"supercategory": "shape", "id": 11, "name": "fire hydrant"},
{"supercategory": "shape", "id": 12, "name": "stop sign"},
{"supercategory": "shape", "id": 13, "name": "parking meter"},
{"supercategory": "shape", "id": 14, "name": "bench"},
{"supercategory": "shape", "id": 15, "name": "bird"},
{"supercategory": "shape", "id": 16, "name": "cat"},
{"supercategory": "shape", "id": 17, "name": "dog"},
{"supercategory": "shape", "id": 18, "name": "horse"},
{"supercategory": "shape", "id": 19, "name": "sheep"},
{"supercategory": "shape", "id": 21, "name": "cow"},
{"supercategory": "shape", "id": 21, "name": "elephant"},
{"supercategory": "shape", "id": 22, "name": "bear"},
{"supercategory": "shape", "id": 23, "name": "zebra"},
{"supercategory": "shape", "id": 24, "name": "giraffe"},
{"supercategory": "shape", "id": 25, "name": "backpack"},
{"supercategory": "shape", "id": 26, "name": "umbrella"},
{"supercategory": "shape", "id": 27, "name": "handbag"},
{"supercategory": "shape", "id": 28, "name": "tie"},
{"supercategory": "shape", "id": 29, "name": "suitcase"},
{"supercategory": "shape", "id": 30, "name": "frisbee"},
{"supercategory": "shape", "id": 31, "name": "skis"},
{"supercategory": "shape", "id": 32, "name": "snowboard"},
{"supercategory": "shape", "id": 33, "name": "sports ball"},
{"supercategory": "shape", "id": 34, "name": "kite"},
{"supercategory": "shape", "id": 35, "name": "baseball bat"},
{"supercategory": "shape", "id": 36, "name": "baseball glove"},
{"supercategory": "shape", "id": 37, "name": "skateboard"},
{"supercategory": "shape", "id": 38, "name": "surfboard"},
{"supercategory": "shape", "id": 39, "name": "tennis racket"},
{"supercategory": "shape", "id": 40, "name": "bottle"},
{"supercategory": "shape", "id": 41, "name": "wine glass"},
{"supercategory": "shape", "id": 42, "name": "cup"},
{"supercategory": "shape", "id": 43, "name": "fork"},
{"supercategory": "shape", "id": 44, "name": "knife"},
{"supercategory": "shape", "id": 45, "name": "spoon"},
{"supercategory": "shape", "id": 46, "name": "bowl"},
{"supercategory": "shape", "id": 47, "name": "banana"},
{"supercategory": "shape", "id": 48, "name": "apple"},
{"supercategory": "shape", "id": 49, "name": "sandwich"},
{"supercategory": "shape", "id": 50, "name": "orange"},
{"supercategory": "shape", "id": 51, "name": "broccoli"},
{"supercategory": "shape", "id": 52, "name": "carrot"},
{"supercategory": "shape", "id": 53, "name": "hot dog"},
{"supercategory": "shape", "id": 54, "name": "pizza"},
{"supercategory": "shape", "id": 55, "name": "donut"},
{"supercategory": "shape", "id": 56, "name": "cake"},
{"supercategory": "shape", "id": 57, "name": "chair"},
{"supercategory": "shape", "id": 58, "name": "couch"},
{"supercategory": "shape", "id": 59, "name": "potted plant"},
{"supercategory": "shape", "id": 60, "name": "bed"},
{"supercategory": "shape", "id": 61, "name": "dining table"},
{"supercategory": "shape", "id": 62, "name": "toilet"},
{"supercategory": "shape", "id": 63, "name": "tv"},
{"supercategory": "shape", "id": 64, "name": "laptop"},
{"supercategory": "shape", "id": 65, "name": "mouse"},
{"supercategory": "shape", "id": 66, "name": "remote"},
{"supercategory": "shape", "id": 67, "name": "keyboard"},
{"supercategory": "shape", "id": 68, "name": "cell phone"},
{"supercategory": "shape", "id": 69, "name": "microwave"},
{"supercategory": "shape", "id": 10, "name": "oven"},
{"supercategory": "shape", "id": 71, "name": "toaster"},
{"supercategory": "shape", "id": 72, "name": "sink"},
{"supercategory": "shape", "id": 73, "name": "refrigerator"},
{"supercategory": "shape", "id": 74, "name": "book"},
{"supercategory": "shape", "id": 75, "name": "clock"},
{"supercategory": "shape", "id": 76, "name": "vase"},
{"supercategory": "shape", "id": 77, "name": "scissors"},
{"supercategory": "shape", "id": 78, "name": "teddy bear"},
{"supercategory": "shape", "id": 79, "name": "hair drier"},
{"supercategory": "shape", "id": 80, "name": "toothbrush"}
]


def loadMP40(filename):
    colortable = [ [0,0,0], [1, 0, 0], [0, 0, 1], 
    [0, 1, 0], [0, 1, 1], [1, 0, 1], 
    [1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 1], 
    [0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5], 
    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], 
    [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
    [0.7, 0, 0], [0, 0.7, 0], [0, 0, 0.7], 
    [0.7, 0.7, 0], [0, 0.7, 0.7], [0.7, 0, 0.7], 
    [0.7, 0.3, 0], [0, 0.7, 0.3], [0.3, 0, 0.7], 
    [0.3, 0.7, 0], [0, 0.3, 0.7], [0.7, 0, 0.3], 
    [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3], 
    [0.3, 0.3, 0], [0, 0.3, 0.3], [0.3, 0, 0.3],
    [1, 0.3, 0.], [0.3, 1, 0.3], [0.3, 0.3, 1], 
    [1, 1, 0.3], [0.3, 1, 1], [1, 0.3, 1], 
    [1, 0.5, 0.3], [0.3, 1, 0.5], [0.5, 0.3, 1], 
    [0.5, 1, 0.3], [0.3, 0.5, 1], [1, 0.3, 0.5], 
    [0.5, 0.3, 0.3], [0.3, 0.5, 0.3], [0.3, 0.3, 0.5], 
    [0.5, 0.5, 0.3], [0.3, 0.5, 0.5], [0.5, 0.3, 0.5],
    [0.3, 0.5, 0.5], [0.5, 0.3, 0.5], [0.5, 0.5, 0.3], 
    [0.3, 0.3, 0.5], [0.5, 0.3, 0.3], [0.3, 0.5, 0.3], 
    [0.3, 0.8, 0.5], [0.5, 0.3, 0.8], [0.8, 0.5, 0.3], 
    [0.8, 0.3, 0.5], [0.5, 0.8, 0.3], [0.3, 0.5, 0.8], 
    [0.8, 0.5, 0.5], [0.5, 0.8, 0.5], [0.5, 0.5, 0.8], 
    [0.8, 0.8, 0.5], [0.5, 0.8, 0.8], [0.8, 0.5, 0.8]  ] 
    
    
    categorydict = {}
    
    with open(filename,"r") as fp:
       lnidx = 0
       for line in fp:
    
          if lnidx==0:
              lnidx+=1
              continue
          tokens = line.split("\t",5)
          rgb = colortable[int(tokens[0])]
          hexstr = "#{0:02x}{1:02x}{2:02x}".format(int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))

          # MP40 ID, MP40 Name, NYU40 Name
          categorydict[hexstr] = (int(tokens[0]),tokens[1],tokens[4])
          
          lnidx+=1
       
	# add remaining ones without class label
    for i in range(lnidx,np.asarray(colortable).shape[0]):

        rgb = colortable[lnidx]
        hexstr = "#{0:02x}{1:02x}{2:02x}".format(int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))

         #print("adding OTH "+hexstr)
        categorydict[hexstr] = (lnidx,"undefined","undefined")
     
        lnidx += 1
   
   
    return (colortable,categorydict)
	
	
def classIdFromColor(rgb,categories):
            
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                hexstr = "#{0:02x}{1:02x}{2:02x}".format(rgb[0]+i,rgb[1]+j,rgb[2]+k)
    
                if hexstr in categories.keys():
                    cat = categories[hexstr]
                    if cat[0]>0:
                        return cat[0]-1
                    else:
                        return 0

    return 0
    
def getNYUClassId(mpname,mpcategories):
    # find NYU name for MP name
    nyuname = "otherprop"
    for entry in mpcategories:
       data = mpcategories[entry]
       if data[1].find(mpname)>-1:
           nyuname = data[2]

	# get NYU ID
    for cat in NYU40_CATEGORIES:
       if cat["name"] == nyuname:
           return cat["id"]
    return 40
	
# return overlapping 
def getCOCOClassId(mpname,mpcategories):
    if mpname=="chair":
        coconame = "chair"
    elif mpname=="table":
        coconame = "dining table"
    elif mpname=="sofa":
        coconame = "couch"
    elif mpname=="bed":
        coconame = "bed"
    elif mpname=="plant":
        coconame = "potted plant"
    elif mpname=="sink":
        coconame = "sink"
    elif mpname=="toilet":
        coconame = "toilet"
    elif mpname=="tv_monitor":
        coconame = "tv"
    else:
        return 0
	
    # get COCO ID
    for cat in COCO_CATEGORIES:
       if cat["name"] == coconame:
           return cat["id"]
    return 0

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_instances(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def generate_color_image_id(file):
    fn = os.path.basename(file).rsplit('/') [0]
    fn = fn.split('.') [0]
    return fn

	
def generate_annotation_id(image_id, instance_id):
    return image_id + '_' + str(instance_id)

def main():

    if CLASS_LABELS == "nyu40":
        CATEGORIES = NYU40_CATEGORIES
    elif CLASS_LABELS == "coco":
        CATEGORIES = COCO_CATEGORIES
	
    MAX_CATEGORIES = len(CATEGORIES)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1 # maps to 1 + encoded scene + image_id
    depth_id = 1 # maps to 2 + encoded scene + depth_id
    segmentation_id = 1 # counter

    (colorTable,categoryTable) = loadMP40(os.path.join(ROOT_DIR,'mpcat40.tsv'))
	
    running_id = 0
	
    iddict = {}

    if opt.do_stats:
        sizelist = []
        multiview_stats = []
        inst_data_dict = {}
							
    # Filter for color jpeg images
    if opt.export_color_images:
	
        for house in HOUSES:
		
            if opt.do_stats:
                inst_data_dict = {}

            cc = os.path.join(ROOT_DIR, SCENE_DIR, house,'matterport_skybox_images')
            for root, _, files in os.walk(cc):
                image_files = filter_for_jpeg(root, files)

                # Go through each color image
                for image_filename in image_files:
                    image = Image.open(image_filename)
                    image_id = generate_color_image_id(image_filename) # FTT
                    
                    file_name = SCENE_DIR + '/' + house + '/' + 'matterport_skybox_images' + '/' + image_id + '.jpg'
                    filenum = os.path.basename(image_filename).rsplit('.')[0]

                    image_info = pycococreatortools.create_image_info(
                        running_id, file_name, image.size)
                    coco_output["images"].append(image_info)
					
                    iddict[running_id] = image_id
					

				
                    # get instance to label mapping
                    mapping_filename = os.path.join(ROOT_DIR, SRC_ANNOTATION_DIR, house, 'sphere_points_smooth', image_id + '_filtered_aggregation.json')
                    if (not os.path.exists(mapping_filename)):
                        print('WARNING, cannot find mapping file', mapping_filename)
                        #sys.exit(1)
                        continue
                    mapping = json.load(open(mapping_filename))
				

                    # Filter for annotation mask file associated with color image and label
                    instance_filename = image_filename.replace('matterport_skybox_images', 'segmentation_maps_instances').replace('.jpg', '.png')
                    print(instance_filename)

                    img = Image.open(instance_filename)
                    pixel = np.array(img)
                    # Go through each colour, as instances are encoded as colour values
                    pixelcolors = [x[1] for x in img.getcolors()]  # [0] = pixelcount, [1] = colour
                    for colortuple in pixelcolors:
                        instance_id = classIdFromColor(colortuple,categoryTable)
                    
                        key = str(instance_id)
                        category_label = ''
                        category_id = 0
                        for seggrp in mapping.get('segGroups'):
                            if int(seggrp.get('id')) == instance_id:
                                category_label = seggrp.get('label')
                                if CLASS_LABELS == "nyu40":
                                    category_id = getNYUClassId(category_label,categoryTable)  # val contains raw and nyu40 label
                                if CLASS_LABELS == "coco":
                                    category_id = getCOCOClassId(category_label,categoryTable)  #  
                        if category_id == 0:  # 0 is background
                            continue
                        if category_id < 1 or category_id > MAX_CATEGORIES:  # Just make sure, we are safe
                            print('Ignore category ' + str(category_id))
                            continue
        
                        # Labels are nyu40id and coded as pixel colours (1 .. 40 decimal)
                        category_info = {'id': category_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.all(pixel == colortuple, axis=-1)  # Create a binary mask for each of the labels
						
                        # use morphology to clean masks 
                        if opt.clean_masks:
                            selem = disk(4)
                            binary_mask = opening(binary_mask,selem)
                            binary_mask = binary_fill_holes(binary_mask)
												
                        if opt.discard_wrap_around_regions > 0:
                            width, height = img.size
                            centerstart = int(width/2 - opt.discard_wrap_around_regions/2)
                            centerend = int(width/2 + opt.discard_wrap_around_regions/2)
						
                            labelled_mask,n_labels = measure.label(binary_mask,return_num =True)
                            labelled_mask_center = labelled_mask[:,centerstart:centerend]
							
                            keep_labels = np.unique(labelled_mask_center)
                            for i in range(1,keep_labels.shape[0]):
                                labelled_mask[labelled_mask == keep_labels[i]] = n_labels+1
								
                            binary_mask = labelled_mask > n_labels

                        # from multiple regions, keep the largest two, if they cross the image border, split
                        labelled_mask,n_labels = measure.label(binary_mask,return_num =True)
                        if n_labels>2:
                            width, height = img.size 
                            centerx = width/2
						
                            props = measure.regionprops(labelled_mask)
                        
                            mostcentralSz = 0
                            largestIdx = 0 
                            largestSz = 0
                            largestIdx2 = 0 
                            largestSz2 = 0
                            idx = 0
                            for prop in props:
                                d = abs(prop.centroid[1] - centerx)
                                if prop.area > largestSz:
                                    largestSz2 = largestSz
                                    largestIdx2 = largestIdx
                                    largestSz = prop.area
                                    largestIdx = idx
                                elif prop.area > largestSz2:
                                    largestSz2 = prop.area
                                    largestIdx2 = idx
								
                                idx = idx+1
						
                            labelled_mask[labelled_mask == largestIdx+1] = n_labels +1
                            labelled_mask[labelled_mask == largestIdx2+1] = n_labels +1
						
                            binary_mask = labelled_mask > n_labels
							
                        labelled_mask,n_labels = measure.label(binary_mask,return_num =True)
                        split = False
                        if n_labels>1 and opt.discard_wrap_around_regions:
                            width, height = img.size
                            centerstart = int(width/2 - opt.discard_wrap_around_regions/2)
                            centerend = int(width/2 + opt.discard_wrap_around_regions/2)

                            #check if labelled content is also outside center
                            labelled_mask_center = copy(labelled_mask)
                            labelled_mask_center[:,centerstart:centerend] = 0
                            remaining = np.unique(labelled_mask_center)
                            if len(remaining)>1:
                                split = True	
                                binary_mask = labelled_mask == 1			
                                binary_mask2 = labelled_mask == 2			

                        # check min size
                        width, height = img.size
                        imgarea = width*height   
                        minrs =  opt.min_region_area * imgarea
                        if np.sum(binary_mask) < minrs:
                            continue
														
                        # store size
                        if opt.do_stats:
                            sizelist.append(float(np.sum(binary_mask))/imgarea)
							

                            if instance_id in inst_data_dict.keys():
                                inst_data_dict[instance_id].append(float(np.sum(binary_mask))/imgarea)
                            else:
							
                                inst_data_dict[instance_id] = [ float(np.sum(binary_mask))/imgarea ]
								
								
                        # debug mask image outputs
                        #maskfile='dbg/mask_' + str(image_id) + '-' + str(instance_id) + '.png'
                        #Image.fromarray((binary_mask * 255).astype(np.uint8)).save(maskfile)
                     
						
                        annotation_id = generate_annotation_id(image_id, instance_id)

                        annotation_info = pycococreatortools.create_annotation_info(
                            annotation_id, running_id, category_info, binary_mask,
                            image.size, tolerance=TOLERANCE)  # 2)
    
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)
							
                        if split:
                            annotation_id = generate_annotation_id(image_id, instance_id + 1000)

                            annotation_info = pycococreatortools.create_annotation_info(
                                annotation_id, running_id, category_info, binary_mask2,
                                image.size, tolerance=TOLERANCE)  # 2)
    
                            if annotation_info is not None:
                                coco_output["annotations"].append(annotation_info)
						
							
                        segmentation_id = segmentation_id + 1

                    running_id = running_id + 1

            if opt.do_stats:
                nMulti = 0
                nMulti005 = 0
                nMulti002 = 0
                nMulti001 = 0
			
                for key in inst_data_dict.keys():
                    inst_data = inst_data_dict[key]
                    inst_ar = np.array(inst_data)
                    inst_ar = inst_ar[inst_ar>0]
                    if len(inst_ar)>1:
                        nMulti = nMulti+1
                        mininst = min(inst_ar)
                        maxinst = max(inst_ar)
						
                        if mininst <= maxinst* 0.01:
                            nMulti001 = nMulti001 + 1
                        elif mininst <= maxinst* 0.02:
                            nMulti002 = nMulti002 + 1
                        elif mininst <= maxinst* 0.05:
                            nMulti005 = nMulti005 + 1

								
                mvdata = [ float(nMulti)/len(inst_data_dict), float(nMulti005)/len(inst_data_dict), float(nMulti002)/len(inst_data_dict), float(nMulti001)/len(inst_data_dict) ]
			
                multiview_stats.append(mvdata)

           
				
    # Filter for depth images (not implemented in this version)
    if opt.export_depth_images:
      print("Depth images: tbd")

    with open(OUTPUT, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
			
    with open(OUTPUT+'.csv', 'w') as f:
        for key in iddict.keys():
            f.write("%d,%s\n"%(key,iddict[key]))
			
    if opt.do_stats:
        print("\nstats:\n\n")
        print("region sizes:\n")
        maxval = 20
        sizelist = np.array(sizelist)
        for i in range(1,maxval,1):
            print("    "+str(i)+": "+str(np.sum(np.logical_and((sizelist>((i-1)/200.0)),(sizelist<(i/200.0))))/len(sizelist)))
        print("  >="+str(maxval)+": "+str(np.sum(sizelist>(maxval/200.0))/len(sizelist)))
        print("\n\nmultiview stats:\n\n")
        print("multiple   0.05   0.02   0.01 ")
        print(np.mean(np.array(multiview_stats),axis=0))
		
if __name__ == "__main__":
    main()
    sys.exit()
