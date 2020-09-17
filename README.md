# matterport_utils
Utilities to prepare data from the [Matterport3D v1 dataset] (https://github.com/niessner/Matterport) for semantic and instance segmentation on panoramic data.

## mpview

A fork of the mpview application, with an additional mode to create the class and instance segmentation maps corresponding to the source views for each of the panoramas.

## preparepano

A script for preparing panoramic data for semantic and instance segmentation, i.e. depth maps, and label maps for classes and instances that are aligned with the provided Matterport skybox images. This results in a set of RGB, depth and label panoramas that can be rendered in a preferred resolution.


Created 2020 by [JOANNEUM RESEARCH] (https://www.joanneum.at) as part of the [ATLANTIS H2020 project] (http://www.atlantis-ar.eu). This work is part of a project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 951900.
