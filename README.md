# crowd_tracker_lidar3d

This is a catkin package for means of preprocessing, annotating and visualizing 3D LiDAR point cloud data. It aims at facilitating LiDAR-based computer vision tasks, such as _object detection_ and _multi-object tracking (MOT)_ with suitable processing tools. Moreover it offers scripts to read and process LiDAR and camera data from rosbag recordings, as well as functions to perform and evaluate point cloud _clustering_. It has been developed in the context of a master project aiming at __People Detection in Close-Proximity for Robot Navigation in Crowds based on 3D LiDAR Data__ at the _Learning Algorithms and Systems Laboratory_ at EPFL. 

The package follows the structure of a ROS catkin python package. In order to work with your own or existing data, create a `\data` folder in the top-level folder. It can contain the point clouds as `.pcd`, `.csv` or `.h5` files. The paths in some of the provided functions might need to be adjusted accordingly. Within the course of the development of this repo, three main datasets have been used and processed: 
1. __Custom Dataset__: a new dataset which has been newly recorded in the course of the master project. The data has been captured on a semi-autonomous wheelchair, named [”qolo”](crowdbot.eu/our-bots/), which is part of the EU-funded project ”crowdbot”. The 3D data was recorded on a single 3D LiDAR sensor, namely the Velodyne VLP-16 at a frame rate of 20Hz in an indoor environment.
2. [LCAS](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-3d-point-cloud-people-dataset/): an indoor point cloud dataset comprising roughly 28K LiDAR scan frames (each containing approximately 30,000 points) recorded on a stationary and moving robot in a university building.
3. [InLiDa](http://simd.albacete.org/datasets/inlida/): another indoor dataset including ~5K annotated LiDAR point cloud frames. 

***
## Package Overview 

### /src 

| Files                        | Purpose |
|------------------------------|---------|
| `src/annotation_utils.py`    | Functions to calculate the heading angle around the upward-facing axis via PCA, as well as the box corners given the parameters of the centroid, the box dimensions and the angle.         |
| `src/hdf5_util.py`           | Utilities to save point clouds to .h5-file format, which will facilitates the saving/loading of point clouds for network training at later stages. |
| `src/loader.py`              | Loader functions to load point cloud data from a csv-file or pcd-file to a numpy array or a pandas dataframe. |
| `src/plot_tools.py`          | Offers different means for visualizing the point clouds, as well as clustering results.  |
| `src/preprocessing.py`       | Functions to standardize/rotate/translate point clouds and means to filter out ground plane.  |
| `src/rosbag_reader.py`       | Class to process rosbag recordings from a Velodyne sensor.      |
| `src/cluster_utils.py`       | Functions to plot and evaluate point cloud clustering results given a specific clustering model. |
<!-- | `src/segmentation.py`        |         | -->
<!-- | `src/optics.py`        |         | -->

### /scripts
| Files                        | Purpose |
|------------------------------|---------|
| `scripts/bounding_box_generator.py`    | Script to generate bounding box annotations from a given point cloud and its centroid. These annotations are then saved to labels for each frame, reading from a directory of point cloud files. |
| `scripts/bounding_box_generator_outlier.py`    | Extends previous scripts to account for outliers in the annotation logic.  |
| `scripts/bounding_box_plotter.py`    | Script to save plots of annotations for means of visual validation of the bounding box annotations for each frame.|
| `scripts/data_to_hdf5.py`    | Script to save point cloud csv-files from qolo-recordings to one hdf5 file per frame/timestamp. |
| `scripts/external_data_to_hdf.py`    | Script to save point cloud csv-files from datasets like _LCAS_ or _InLiDa_ to hdf5 file per frame. |
| `scripts/file_annotator.py`    | Automatic annotation for the _static_ qolo recordings outputting the centroid of the person in the recorded frame.|
| `scripts/rosbag_reader.py`    | Script to read from a specific rosbag or a directory of several rosbags and extract LiDAR and/or camera data into a given file format. |
| `scripts/LCAS_bbox_generator.py`  | Generates bounding box labels, as defined by _(centroid, h,w,l, angle)_ for the LCAS dataset given their specific label annotation format. Boudning boxes are saved to `.h5` files. |
| `scripts/LCAS_to_hdf.py`    | Script reading LCAS data from csv files and saving point clouds to `.h5`-files.|

### /jupyter_nbs -- jupyter notebooks 
The jupyter notebooks provided in this repo offer an interactive mean to manipulate point clouds, test various clustering approaches and to directly visualize them in either 3D interactive plots, or 2D projections. 

| Files                        | Purpose |
|------------------------------|---------|
| `jupyter_nbs/PointCloud_Segmentation.py`    | This notebook explores on unsupervised point cloud clustering techniques, such as DBSCAN and OPTICS. |
| `jupyter_nbs/VisualValidation.py`    | This notebook visualizes and qualitatively evaluates the (semi-)automatically generated bounding box annotations for new raw point cloud recordings. |







***
