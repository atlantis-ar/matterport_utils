# matterport_coco

Script to create COCO style annotations from Matterport instance labels, using either NYU40 or COCO labels.

Note that COCO labels use linear numbering for the 80 classes.

Requires [pycococreatortools](https://github.com/waspinator/pycococreator/tree/master/pycococreatortools)

## Usage

```
--matterport_root_dir          input path to root directory (up to the /v1/ directory of the Matterport dataset
--matterport_house_id          id of the house(s) to be processed (multiple space separated IDs may be provided)
--matterport_scene_dir         input directory for color images (e.g., matterport_skybox_images)
--matterport_annotation_dir    directory containing ScanNet style annotation files
--coco_annotation_dir          output root for annotations
--coco_annotation_file		   filename for annotation JSON file
--tolerance                    tolerance for mask smoothing, higher is smoother (default: 2)
--class_labels                 nyu40 or coco

advanced options:

--discard_wrap_around_regions  if >0, remove regions not entirely included in the center part of the specified width (for images that wrap around)
--clean_masks                  perform morphological operations and hole filling on masks
--min_region_area              require min size of region to be kept (given as fraction of the image area)
--do_stats                     calculate statistics on object sizes and recurrent objects in other views

```

## Examples

```
<<<<<<< HEAD
# one house with COCO labels
=======
# one houses with COCO labels
>>>>>>> d20f78b8f005b34e3c63c740de9f8e3661f9f4c0

matterport-to-coco.py --matterport_root_dir datasets/Matterport/v1 --matterport_scene_dir equirect --matterport_annotation_dir ply --coco_annotation_dir datasets/Matterport/v1/coco_format --coco_annotation_file matterport_test_coco.json --class_labels coco --matterport_house_id 2t7WUuJeko7 
```

```
# all test houses with NYU40 labels

matterport-to-coco.py --matterport_root_dir datasets/Matterport/v1 --matterport_scene_dir equirect --matterport_annotation_dir ply --coco_annotation_dir datasets/Matterport/v1/coco_format --coco_annotation_file matterport_test_nyu40.json --class_labels nyu40 --matterport_house_id 2t7WUuJeko7 5ZKStnWn8Zo ARNzJeq3xxb fzynW3qQPVF jtcxE69GiFV pa4otMbVnkk q9vSo1VnCiC rqfALeAoiTq UwV83HsGsw3 wc2JMjhGNzB WYY7iVyf5p8 YFuZgdQ5vWj yqstnuAEVhm YVUC4YcDtcY gxdoqLR6rwA gYvKGZ5eRqb RPmz2sHmrrY Vt2qJdWjCF2
```


Created 2020 by [JOANNEUM RESEARCH](https://www.joanneum.at) and [CERTH ITI](https://www.iti.gr/iti/index.html) as part of the [ATLANTIS H2020 project](http://www.atlantis-ar.eu). This work is part of a project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 951900.
