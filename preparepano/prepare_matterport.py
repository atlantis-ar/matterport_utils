# Created 2020 by JOANNEUM RESEARCH as part of the ATLANTIS H2020 project
# https://www.joanneum.at
# http://www.atlantis-ar.eu
#
# This tool is part of a project that has received funding from the European 
# Union's Horizon 2020 research and innovation programme under grant 
# agreement No 951900.


import argparse
import os
import sys
import numpy as np
from PIL import Image
# conversion package for panoramic images
# https://github.com/sunset1995/py360convert
# can be installed using pip install py360convert
import py360convert 
import zipfile
import createpano
import logging
import tqdm
import copy
import math

log = logging.getLogger(__name__)

def unzip(basedir,filename):
    with zipfile.ZipFile(os.path.join(basedir, filename), 'r') as zip_ref:
        zip_ref.extractall(basedir)

def parse_camera_params(filename: str) -> dict:
    with open(filename, 'r') as f:
        paramdict = {}
        while True: 
            line = f.readline() 
            if not line: 
                break
            lineparts = line.split(" ")
            if lineparts[0]=="scan":
                (loc,row,ori) = lineparts[1].split("_", 3)
                rowid = int(row[1:])
                ori, _ = os.path.splitext(ori)
                oriid = int(ori)                
                trmatrix = np.array([
                    [float(lineparts[3]), float(lineparts[4]), float(lineparts[5]), float(lineparts[6])],
                    [float(lineparts[7]), float(lineparts[8]), float(lineparts[9]), float(lineparts[10])],
                    [float(lineparts[11]), float(lineparts[12]), float(lineparts[13]), float(lineparts[14])],
                    [float(lineparts[15]), float(lineparts[16]), float(lineparts[17]), float(lineparts[18])]
                ])
                if not(loc in paramdict.keys()):
                    paramdict[loc] = {}
                if not(rowid in paramdict[loc].keys()):
                    paramdict[loc][rowid] = {}
                paramdict[loc][rowid][oriid] = trmatrix
    return paramdict


def correct_depth_distortion(depth_img_in):
    depth_img = copy.deepcopy(depth_img_in)
    
    c1 = depth_img.shape[1]/2
    c0 = depth_img.shape[0]/2
    halfFov = createpano.default_fov / 2
    
    for i in range(depth_img.shape[1]):
        angle1 = (abs(i-c1)/c1) * halfFov
        for j in range(depth_img.shape[0]):
            d1 = math.tan(angle1) * depth_img[j,i,0]
            angle0 = (abs(j-c0)/c0) * halfFov    
            d0 = math.tan(angle0) * depth_img[j,i,0] 
            diag = math.sqrt(d0*d0 + d1*d1)
            distToCenter = math.sqrt(float(depth_img[j,i,0])*float(depth_img[j,i,0]) + diag*diag)
            corr = distToCenter - depth_img[j,i,0]
            
            if depth_img[j,i,0] >0:
                result = depth_img[j,i,0] + corr
                if result>65535:
                    result = 65535
                depth_img[j,i,0] = result

        
    return depth_img
    

def process_file_type(
    base_dir: str,
    scan_id: str,
    file_type: str,
    name: str,
    out_dir: str,
    extension: str,
    is_skyBox: bool,
    interpolate: bool,
    warp_depth: bool
) -> None:
    face_seq = ['U','B','R','F','L','D']
    
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if not(os.path.exists(os.path.join(out_dir, name))):
        os.mkdir(os.path.join(out_dir, name))
    
    srcdir = os.path.join(base_dir, scan_id, scan_id, name)
    filelist = sorted(os.listdir(srcdir))    
    filedict = {}
    for filename in filelist:
        if not(filename.endswith(extension)):
            continue
        namepart, _ = os.path.splitext(filename)
        tokens = namepart.split("_", 3)        
        srcimg = np.array(Image.open(os.path.join(srcdir, filename)))
        if srcimg.ndim==2:
            srcimg = np.reshape(srcimg, (srcimg.shape[0],srcimg.shape[1],1))		
        locationId = tokens[0]
        if is_skyBox:
            if not(locationId) in filedict.keys():
                filedict[locationId] = {}
            filedict[locationId][face_seq[len(filedict[locationId])]] = srcimg
        else:
            if not(locationId) in filedict.keys():
                filedict[locationId] = [ srcimg ]
            else:
                filedict[locationId].append(srcimg)
        
    if not is_skyBox:
        paramdict = parse_camera_params(os.path.join(base_dir, scan_id, scan_id, "undistorted_camera_parameters", scan_id + ".conf"))
            
    for location in tqdm.tqdm(filedict.keys(), desc=f"{file_type}"):
        if is_skyBox:
            facelist = [
                np.fliplr(filedict[location]['F']),
                filedict[location]['R'],
                filedict[location]['B'],
                np.fliplr(filedict[location]['L']),
                filedict[location]['U'],
                np.flipud(filedict[location]['D']) 
            ]
            eqrar = py360convert.c2e(facelist, equirect_size[1], equirect_size[0], mode='bilinear', cube_format='list')
            eqrar = np.fliplr(eqrar)
            eqrimg = Image.fromarray(eqrar.astype(np.uint8))            
            eqrimg.save(os.path.join(out_dir, name, location + ".png"))
        else:
            v = createpano.get_angles(paramdict[location])
            blending = True
            if name.startswith("segmentation_maps"):
                blending = False

            is_depth = False
            if name == "undistorted_depth_images":
                is_depth = True
                blending = False
                if warp_depth:
                    for i in range(len(filedict[location])):                       
                        depth_img= correct_depth_distortion(filedict[location][i])           

                        # debug code
                        #array_buffer = depth_img.astype(np.uint16).tobytes()
                        #eqrimg = Image.new("I", (depth_img.shape[1],depth_img.shape[0]))
                        #eqrimg.frombytes(array_buffer, 'raw', "I;16")               
                        #eqrimg.save(os.path.join(out_dir, name, location + "_" + str(i) + "_corrected.png"), "PNG", compress_level=0)

                        
                        filedict[location][i] = depth_img

            eqrar = createpano.combine_views(filedict[location], v, equirect_size, blending, is_depth)
            if name=="undistorted_depth_images":
                array_buffer = eqrar.astype(np.uint16).tobytes()
                eqrimg = Image.new("I", (eqrar.shape[1],eqrar.shape[0]))
                eqrimg.frombytes(array_buffer, 'raw', "I;16")               
                eqrimg.save(os.path.join(out_dir, name, location + ".png"), "PNG", compress_level=0)
            elif name.startswith("segmentation_maps"):
                eqrimg = Image.fromarray(eqrar.astype(np.uint8))
                eqrimg.save(os.path.join(out_dir, name, location + ".png"), "PNG", compress_level=0)
            else:
                eqrimg = Image.fromarray(eqrar.astype(np.uint8))
                eqrimg.save(os.path.join(out_dir, name, location + ".png"))

def process_scan(m3d_path, out_path, scan_id, types, unpack, warp_depth) -> None:      
    if unpack:
        unzip(os.path.join(m3d_path,scan_id),"undistorted_camera_parameters.zip")
        unzip(os.path.join(m3d_path,scan_id),"house_segmentations.zip")
        unzip(os.path.join(m3d_path,scan_id),"undistorted_color_images.zip")
        unzip(os.path.join(m3d_path,scan_id),"undistorted_depth_images.zip")
        unzip(os.path.join(m3d_path,scan_id),"matterport_skybox_images.zip")

    equirect_path = os.path.join(out_path, scan_id)
    for t in tqdm.tqdm(types, desc="Scan Progress"):
        args = _CHOICE_MAPPING_[t]
        process_file_type(m3d_path, scan_id, t, args[0], equirect_path, args[1], args[2], args[3], warp_depth)

_CHOICE_MAPPING_ = {
    # choice:   (         `folder`,             'ext'   'sky?`  `bilinear`)
    'skybox':   ('matterport_skybox_images',    'jpg',  True,   True),
    'color':    ('undistorted_color_images',    'jpg',  False,  True),
    'depth':    ('undistorted_depth_images',    'png',  False,  True),
    'classes':  ('segmentation_maps_classes',   'png',  False,  False),
    'instances':('segmentation_maps_instances', 'png',  False,  False),
}

def parse_arguments(args):
    usage_text = (
        "Matterport3D preprocessing script"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("--m3d_path", type=str, 
        help="Input Matterport3D root path"
    )
    parser.add_argument("--out_path", type=str,         
        help="Output processed Matterport3D equirectangular images path"
    )
    parser.add_argument("--out_width", type=int, 
        default=1024, help="Output equirectangular width"
    )
    parser.add_argument("--types", #type=list,
        nargs='+', default=['color'],
        choices=['skybox', 'color', 'depth', 'classes', 'instances'],
        help="Which type of files to convert"
    )
    parser.add_argument("--warp_depth", type=bool, default=True,
        help="Apply correction of depth maps to represent radius"
    )
    parser.add_argument("--scan_id", type=str,
        help="Process single specified scan rather than getting list of all scans"
    )
    parser.add_argument("--all_test_scans", action="store_true",
        help="Process all scans of the test set rather than getting list of all scans"
    )
    parser.add_argument("--unpack", action="store_true", 
        help="Unpack ZIP files before processing"
    )
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, _ = parse_arguments(sys.argv)
    equirect_size = [args.out_width, args.out_width // 2]
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    scan_id_list = []
    if not(args.scan_id==None):
        scan_id_list = tqdm.tqdm([args.scan_id], desc="Dataset Progress")
    elif args.all_test_scans:
        test_id_list = [
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
        scan_id_list = tqdm.tqdm(test_id_list, desc="Dataset Progress")
    else: 
        scan_id_list = tqdm.tqdm(os.listdir(args.m3d_path), desc="Dataset Progress")
    for scan_id in scan_id_list:
        process_scan(args.m3d_path, args.out_path, scan_id, args.types, args.unpack, args.warp_depth)
