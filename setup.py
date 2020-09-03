
#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['crowd_tracker_lidar3d'],
    package_dir={'': 'src'},
    # scripts=['scripts/rosbag_reader', 
    #          'scripts/file_annotator',
    #          'scripts/bounding_box_generator']
)

setup(**d)
