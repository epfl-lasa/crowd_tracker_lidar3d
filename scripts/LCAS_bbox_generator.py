#!/usr/bin/env python
'''
    Adds approximate minimum 3D bounding box to data labels. Encoded with 6 DOF, 
    i.e. position of box centroid (x,y,z) as well as box dimensions (h,w,l) and 
    heading angle (ry) measured around y axis.
'''

import numpy as np
import math
import os
import pandas as pd
import string
import scipy
import shutil

from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crowd_tracker_lidar3d.loader import load_data_to_dataframe
from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord
from crowd_tracker_lidar3d.hdf5_util import save_h5_basic, load_h5_basic
from crowd_tracker_lidar3d.annotation_utils import calc_heading_angle, boxes3d_to_corners3d_velodyne

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/LCAS/hdf5/")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/LCAS/annotated_boundbox/")
LCAS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/LCAS/")
name_to_category = {'pedestrian': 0, 'group': 1}


# fg_pt_flag = in_hull(pts_coor, box_corners)
def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)
    return flag

def get_boxes_from_labels(labels, data): 
    bboxes = []
    pts_coor = data[:,:3]
    for label in labels: 
        # category = name_to_category[label[0]]
        category = label[0]
        if category == 'wheelchair':
            continue
        print(category)
        centroid = np.array((float(label[1]), float(label[2]), float(label[3])))
        min_x, min_y, min_z = float(label[4]), float(label[5]), float(label[6])
        max_x, max_y, max_z = float(label[7]), float(label[8]), float(label[9])
        visibility = float(label[10]) # 0 = visible, 1 = partially visible 
        # create bounding box parameters (h,w,l)
        h = max_z - min_z # assume that legs always on ground  
        w = max_x - min_x
        l = max_y - min_y

        # enlarge bbox by a small constant 
        const = 0.1
        h = h + const 
        w = w + const 
        l = l + const 
        
        bbox = np.concatenate((centroid, (h,w,l), ([0.0]))) # preliminary angle=0
        bbox_temp = np.reshape(bbox, (-1,7))
        box_corners = np.reshape(boxes3d_to_corners3d_velodyne(bbox_temp, rotate=False), (8,3))

        # find points which are inside box 
        pts_flag = in_hull(pts_coor, box_corners)
        if np.sum(pts_flag) >0: 
            box_points = pts_coor[pts_flag]
            # Compute orientation 
            bbox[6], _ = calc_heading_angle(box_points)
        
        bboxes.append(bbox)
    return np.array(bboxes)


def main(): 
    print(DATA_DIR)
    # iterate over folders, each containing a data recording scene
    for folder in sorted(os.listdir(DATA_DIR)):
        print("------------------ Sequence: {} ------------------\n".format(folder))
        path = os.path.join(DATA_DIR, folder) 
        out_dir = os.path.join(SAVE_DIR, folder) #save frame files in separate directory
        os.makedirs(out_dir,exist_ok=True)
        label_folder = folder+'_labels'
        idx_map_file = os.path.join(path, 'idx_map.txt')
        idx_map = {}
        with open(idx_map_file, 'r') as file:
            lines = file.readlines()
            for line in lines: 
                idx, name = line.strip().split(':')
                idx_map[int(idx)] = name
        data_files = [str(f) for f in os.listdir(path) if f.endswith('.h5')] 
        
        # Save data per timeframe
        out_dir = os.path.join(SAVE_DIR, folder) #save files in separate directory
        os.makedirs(out_dir, exist_ok=True)
        print("Processing {} frames".format(len(data_files)))
        
        annotated_frames_file = os.path.join(out_dir, 'annotated_frames.txt')
        with open(annotated_frames_file, 'w') as out:
            # iterate over labeled point cloud frames saved in hdf5 files 
            empty_frames = 0 
            annotated_frames = 0 
            total_boxes = 0 
            boxes_with_angle = 0 
            for idx, frame in enumerate(data_files): 
                full_file = os.path.join(path, frame)
                data = load_h5_basic(full_file) 

                print('===== {} ====='.format(frame))
                print('Number of points: {}'.format(data.shape[0]))

                frame_name = frame.split('.')[0]
                # print(frame_name)
                frame_idx = int(frame_name.replace('frame',''))
                print(frame_idx)
            
                # get bounding box from label 
                labels = []

                # check if frame annotated
                label_path = os.path.join(LCAS_BASE, label_folder, idx_map[frame_idx]+'.txt')
                if os.path.exists(label_path):
                    out.write('{}:{}\n'.format(frame_idx, idx_map[frame_idx]))
                    annotated_frames += 1 
                    # read labels 
                    with open(label_path, 'r') as label_file:
                        lines = label_file.readlines()
                        # print(lines)
                        for line in lines: 
                            labels.append([item for item in line.strip().split(' ')])

                    bboxes = get_boxes_from_labels(labels, data)
                    for box in bboxes: 
                        total_boxes += 1
                        if box[6] != 0: 
                            boxes_with_angle +=1
                    if bboxes.shape[0] == 0: 
                        empty_frames += 1
                        continue
                    print('Bounding Box List - Shape: {}'.format(bboxes.shape))
                    print('Saved annotations under: {}'.format(os.path.join(out_dir,frame)))
                    save_h5_basic(os.path.join(out_dir,frame), bboxes)
                else: 
                    print('No annotation') 
            print("\n\n{}/{} frames empty.".format(empty_frames, len(data_files)))
            print('{}/{} annotated.'.format(annotated_frames, len(data_files)))
            print('{}/{} boxes with angle.'.format(boxes_with_angle, total_boxes))

if __name__=='__main__':
    main()