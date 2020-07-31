#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import os 
import numpy as np 
import shutil

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from crowd_tracker_lidar3d.hdf5_util import save_h5, load_h5
from crowd_tracker_lidar3d.annotation_utils import calc_heading_angle

'''
Visual validation of boudning box annotations for each frame via several plotting functions.
'''

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/annotated_boundbox/")
WALK = False 

def save_fig(fig, path, extension='.pdf'):
    """ Helper Function to save plots 

    Args:
        fig: matplotlib figure instance
        path (str): path to save the figure to 
        extension (str, optional): Define with which extension the file should be saved. Defaults to '.pdf'.
    """
    save_dir = path + extension
    fig.savefig(save_dir, bbox_inches='tight')  

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=1,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def calc_3d_box(bbox): 
    x, y, z, h, w, l = bbox[:6]
    # h -= 0.1
    # w -= 0.1
    # l -= 0.1
    box8 = np.array(
        [
            [
                x + w / 2,
                x + w / 2,
                x - w / 2,
                x - w / 2,
                x + w / 2,
                x + w / 2,
                x - w / 2,
                x - w / 2,
            ],
            [
                y - l / 2,
                y + l / 2,
                y + l / 2,
                y - l / 2,
                y - l / 2,
                y + l / 2,
                y + l / 2,
                y - l / 2,
            ],
            [
                0,
                0,
                0,
                0,
                z + h/2,
                z + h/2,
                z + h/2,
                z + h/2,
            ],
        ]
    )
    return box8.T

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

def display_bboxes_in_data(dataset, label, gt_boxes3d, axes_limits, plot=True, angle=False, bbox=None):
    """
    Displays the (ground truth) bounding boxes corresponding to the detection/label within the scatter 
    plot of the LiDAR data within 4 plots (one in 3D, and 3 projections).
    
    Parameters
    ----------
    dataset         : `raw` dataset 
    gt_boxes3d      : numpy nd array with bbox defined by its 8 vertices (in 3d spatial coordinates)
    """
    
    axes_str = ['X', 'Y', 'Z']
    template_mask = np.where(label)
    background_mask = np.where(~label)

    def draw_point_cloud(ax, title, template_mask, background_mask, axes=[0, 1, 2], xlim3d=None, 
                         ylim3d=None, zlim3d=None, plot=plot):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
            # Size of point markers in plots
        point_size = 0.5
        point_size_template = 2.0
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Positive label', markerfacecolor='r', markersize=10)
            ]
        template = dataset[template_mask]
        background = dataset[background_mask]
        ax.scatter(*np.transpose(background[:, axes]), s=point_size, c=background[:, 3], cmap='viridis')
        ax.scatter(*np.transpose(template[:, axes]), s=point_size_template, c='r')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        ax.legend(handles=legend_elements, loc='upper right')

        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
#         User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
        
        num = len(gt_boxes3d)
        for n in range(num):
            b = gt_boxes3d[n].T
            draw_box(ax, b, axes=axes, color='r')
            
    # # Draw point cloud data as 3D plot
    # f2 = plt.figure(figsize=(10, 8))
    # ax2 = f2.add_subplot(111, projection='3d')                    
    # draw_point_cloud(ax2, 'Velodyne scan', xlim3d=axes_limits[2])
    # if plot:
    #     plt.show()
    
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(3, 1, figsize=(10, 15))
    draw_point_cloud(
        ax3[0], 
        'Velodyne scan, XZ projection (Y = 0)', 
        template_mask,
        background_mask,
        axes=[0, 2] # X and Z axes
    )
    draw_point_cloud(
        ax3[1], 
        'Velodyne scan, XY projection (Z = 0)', 
        template_mask,
        background_mask,
        axes=[0, 1] # X and Y axes
    )
    if (angle) & (type(bbox)== np.ndarray):
        _, pca_stats = calc_heading_angle(dataset, template_mask)
        v = pca_stats['components'][0]
        centroid = bbox[0:2]
        draw_vector(centroid, centroid + v, ax=ax3[1])

    draw_point_cloud(
        ax3[2], 
        'Velodyne scan, YZ projection (X = 0)', 
        template_mask,
        background_mask,
        axes=[1, 2] # Y and Z axes
    )
    if plot: 
        plt.show()
    return f

def main(): 
    for folder in sorted(os.listdir(DATA_DIR)):
        if WALK: 
            start = 'walk'
        else: 
            start = 'person'
            
        if folder.startswith(start):
            print('---------------------')
            print(folder)
            path = os.path.join(DATA_DIR, folder) 
            data_files = [str(f) for f in sorted(os.listdir(path)) if f.endswith('.h5')] 
            dist = int(folder.split('_')[-1])
            
            # create subdirectory to save plots 
            SAVE_DIR = os.path.join(path, 'plots')
            if os.path.exists(SAVE_DIR):
                shutil.rmtree(SAVE_DIR)
                os.makedirs(SAVE_DIR)
            #     pass # File has already been processed
            else:
                os.makedirs(SAVE_DIR)
                
            # downsample frames 
            # frame_sample = data_files[::20]
            frame_sample = sorted(data_files)[20:40]
            print('Validating {}/{} frames'.format(len(frame_sample), len(data_files)))
            
            for frame in frame_sample: 
                print(frame)
                # load data 
                f = os.path.join(path, frame)
                data, label, bbox = load_h5(f, bbox=True)
                centroid, h, w, l, angle = bbox[0:3], bbox[3], bbox[4], bbox[5], bbox[6]
                mask = np.where(label)
                template = data[mask]
                # calculate bounding box on labeled data 
                box8 = calc_3d_box(bbox)
                gt_box = np.reshape(box8, (-1,8,3))

                if not WALK:           # static datasets 
                    axes_limits = [
                        [dist-2, dist+2], # X axis range
                        [-1, 2], # Y axis range
                        [-0.5, 3]   # Z axis range
                    ]
                else: 
                    # walking datasets
                    x_center, y_center =centroid[0:2]
                    axes_limits = [
                        [x_center-3, x_center+3], # X axis range
                        [y_center-2, y_center+2], # Y axis range
                        [-0.5, 3]   # Z axis range
                    ]
                
                detections_proj = display_bboxes_in_data(data, label, gt_box, axes_limits, angle=True, bbox=bbox, plot=False)
                frame_name = str.rstrip(frame, '.h5')
                save_fig(detections_proj, os.path.join(SAVE_DIR, frame_name), extension='.pdf')
                plt.close('all')
            # break

if __name__=='__main__':
    main()
