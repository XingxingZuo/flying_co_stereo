# depth_align package
## Description
This package is used for aligning sparse metric landmarks with relative depth predicted by DepthAnythingV2.
## Usage
To launch the depth alignment node, use the following command:
```bash
roslaunch depth_align align_single_depth.launch
```
To visualize the predicted PointCloud, use the following command:
```bash
cd DepthAlign
rviz -d rviz/visualize.rviz
```