# preparepano

Script to prepare panoramic data for semantic and instance segmentation from the [Matterport3D v1 dataset](https://github.com/niessner/Matterport).

## prepare_matterport

The script prepares data for segmentation, requiring the following inputs from for a Matterport scan:

- undistorted camera parameters
- house segmentations
- undistorted color images
- undistorted depth images
- Matterport skybox images

In addition, the class and instance segmentation maps created using the modified version of [mpview](https://github.com/atlantis-ar/matterport_utils/tree/master/mpview) are needed, expected in segmentation_maps_classes and segementation_maps_instances directories of the Matterport scene.

It creates the following aligned outputs (as equirectangular images):
- Matterport skybox images tranformed from cubemaps to equirectangular
- RGB panorama from undistorted color images
- depth panorama
- class label panorama
- instance label panorama

## createpano

(used by prepare_matterport)

The basis of the implementation is ported from [PanoBasic](https://github.com/yindaz/PanoBasic) written in MATLAB. The code has been extended to cover the specific requirements for also merging depth maps and segementation label maps to a panorama.



Created 2020 by [JOANNEUM RESEARCH](https://www.joanneum.at) as part of the [ATLANTIS H2020 project](http://www.atlantis-ar.eu). This work is part of a project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 951900.