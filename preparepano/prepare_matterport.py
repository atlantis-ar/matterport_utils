import argparse
import os
import sys
import numpy as np
from PIL import Image
# conversion package for panoramnic images
# https://github.com/sunset1995/py360convert
# can be installed using pip install py360convert
import py360convert 
import zipfile
import createpano
import logging
import tqdm

log = logging.getLogger(__name__)

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

def process_file_type(
    base_dir: str,
    scan_id: str,
    file_type: str,
    name: str,
    out_dir: str,
    extension: str,
    is_skyBox: bool,
    interpolate:bool
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
            eqrimg.save(os.path.join(out_dir, name, location + ".jpg"))
        else:
            v = createpano.get_angles(paramdict[location])
            blending = True
            if name.startswith("segmentation_maps"):
                blending = False
            eqrar = createpano.combine_views(filedict[location], v, equirect_size, blending)
            if name=="undistorted_depth_images":
                array_buffer = eqrar.astype(np.uint16).tobytes()
                eqrimg = Image.new("I", (eqrar.shape[1],eqrar.shape[0]))
                eqrimg.frombytes(array_buffer, 'raw', "I;16")
                #eqrimg = Image.fromarray(eqrar.astype(np.uint16))                
                eqrimg.save(os.path.join(out_dir, name, location + ".png"), "PNG", compress_level=0)
            elif name.startswith("segmentation_maps"):
                eqrimg = Image.fromarray(eqrar.astype(np.uint8))
                eqrimg.save(os.path.join(out_dir, name, location + ".png"), "PNG", compress_level=0)
            else:
                eqrimg = Image.fromarray(eqrar.astype(np.uint8))
                eqrimg.save(os.path.join(out_dir, name, location + ".jpg"))

def process_scan(m3d_path, out_path, scan_id, types) -> None:      
    equirect_path = os.path.join(out_path, scan_id)
    for t in tqdm.tqdm(types, desc="Scan Progress"):
        args = _CHOICE_MAPPING_[t]
        process_file_type(m3d_path, scan_id, t, args[0], equirect_path, args[1], args[2], args[3])

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
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, _ = parse_arguments(sys.argv)
    equirect_size = [args.out_width, args.out_width // 2]
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    for scan_id in tqdm.tqdm(os.listdir(args.m3d_path), desc="Dataset Progress"):
        process_scan(args.m3d_path, args.out_path, scan_id, args.types)
