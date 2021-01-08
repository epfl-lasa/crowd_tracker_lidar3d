import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


def plot_pointcloud3d(data, point_size=None): 
    """ Generate a matplotlib 3D plot of the given point cloud data. 

    Args:
        data (pandas DataFrame or numpy  ndarray): columns of df should contain at least contain 'x', 'y', 'z' 
        point_size (float): Parameter to adjust the size of each plotted point. Defaults to None.
    """
    if not point_size: 
        no_points = data.shape[0]
        point_size = 10**(3- int(np.log10(no_points))) # Adjust point size based on point cloud size

    plt.ion()
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    if isinstance(data, pd.DataFrame):
        ax.scatter(data['x'], data['y'], data['z'], c=data['intensity'], s=point_size*5, edgecolor='', marker='o')
    else: 
        if data.shape[1] == 4:
            ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], s=point_size*5, edgecolor='', marker='o')
        elif data.shape[1] == 3:  # If point cloud is XYZ format 
            ax.scatter(data[:,0], data[:,1], data[:,2], s=point_size*5, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.title('3D Point Cloud data')
    plt.show()
    
def draw_point_cloud(data, ax, title, axes=['x', 'y', 'z'], axes_limits=None, xlim3d=None, ylim3d=None, zlim3d=None, point_eliminations=None):
    """ Plot given point cloud either as a projection in 2D or as 3D plot. 

    Args:
        data (pandas DataFrame): columns of df should contain at least contain 'x', 'y', 'z'
        ax ([matplotlib axis]): the axis to plot on 
        title (str): title of the plot/figure
        axes (list, optional): The axes to project the data on. Defaults to ['x', 'y', 'z'].
        axes_limits (list, optional): Defines limits for each axis. Defaults to None.
        point_eliminations (pandas DataFrame, optional): Points to color in red, because they have been classified as outliers. Defaults to None.

    Returns:
        [type]: [description]
    """
    no_points = data.shape[0]
    point_size = 10**(3- int(np.log10(no_points))) # Adjust point size based on point cloud size
    
    if data.shape[1] == 4: # If point cloud is XYZI format (I = intensity)
        im = ax.scatter(*np.transpose(data[axes].to_numpy()), s = point_size, c=data['intensity'], cmap='viridis')
        if point_eliminations is not None: 
            ax.scatter(*np.transpose(point_eliminations[axes].to_numpy()), s = point_size, c='r', alpha = 0.7)

    elif data.shape[1] == 3:   # If point cloud is XYZ format 
        im = ax.scatter(*np.transpose(data[axes].to_numpy()), s = point_size, c='b', alpha = 0.3)
        if point_eliminations is not None: 
            ax.scatter(*np.transpose(point_eliminations[axes].to_numpy()), s = point_size, c='r', alpha = 0.7)

    ax.set_xlabel('{} axis'.format(axes[0]), fontsize=16)
    ax.set_ylabel('{} axis'.format(axes[1]), fontsize=16)
    
    if not axes_limits:
        x_max, x_min = np.max(data.x), np.min(data.x)
        y_max, y_min = np.max(data.y), np.min(data.y)
        z_max, z_min = np.max(data.z), np.min(data.z)
        
        axes_limits = [
            [x_min, x_max], # X axis range
            [y_min, y_max], # Y axis range
            [z_min, z_max]   # Z axis range
        ]
    
    if len(axes) > 2: # 3-D plot
        ax.set_xlim3d(axes_limits[0])
        ax.set_ylim3d(axes_limits[1])
        ax.set_zlim3d(axes_limits[2])
        ax.set_zlabel('{} axis'.format(axes[2]), fontsize=16)
        
    else: # 2-D plot
        ax.set_xlim(*axes_limits[0])
        ax.set_ylim(*axes_limits[1])
    
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)
    
    ax.set_title(title, fontsize=18)
    return im 

def show_projections(raw_data, dimensions=None, savefig=False, filename=None, point_eliminations=None): 
    """
    Creates 3 plots of 3 different projections of the given point cloud, i.e. xy-/yz-/ and xz-pojection. 

    Args:
        raw_data (pandas DataFrame): the raw data loaded from rosbags to a dataframe
        dimensions (list): containing the column names (string) of the dataframe 
                           (namely x,y,z), if 'intensity' is included, points are colored 
                           according to their intensity value
        savefig (bool, default=False)
    """
    if dimensions is not None: 
        data = raw_data[dimensions]
    else: 
        data = pd.DataFrame(raw_data, columns=['x', 'y', 'z', 'intensity'])

    if point_eliminations is not None:
        point_eliminations = point_eliminations[dimensions]
    fig, ax3 = plt.subplots(1, 3, figsize=(20, 10)) # plots in 3 columns
    # f, ax3 = plt.subplots(3, 1, figsize=(12, 25)) if plots in 1 column 

    x_max, x_min = np.max(data.x), np.min(data.x)
    y_max, y_min = np.max(data.y), np.min(data.y)
    z_max, z_min = np.max(data.z), np.min(data.z)

    if point_eliminations is not None: 
        x_max, x_min = np.max(list(data.x) +  list(point_eliminations.x)), np.min(list(data.x) + list(point_eliminations.x))
        y_max, y_min = np.max(list(data.y) + list(point_eliminations.y)), np.min(list(data.y) + list(point_eliminations.y))
        z_max, z_min = np.max(list(data.z) + list(point_eliminations.z)), np.min(list(data.z) + list(point_eliminations.z))
            
        
    axes_limits = [
        [x_min, x_max], # X axis range
        [y_min, y_max], # Y axis range
        [z_min, z_max]   # Z axis range
    ]

    im1 = draw_point_cloud(data,
            ax3[0], 
            'XZ projection (Y = 0)', 
            axes=['x', 'z'], # X and Z axes,
            axes_limits=[axes_limits[0], axes_limits[2]],
            point_eliminations=point_eliminations
        )

    im2 = draw_point_cloud(data,
            ax3[1], 
            'XY projection (Z = 0)', 
            axes=['x', 'y'], # X and Y axes
            axes_limits=[axes_limits[0], axes_limits[1]],
            point_eliminations=point_eliminations
        )

    im3 = draw_point_cloud(data,
            ax3[2], 
            'YZ projection (X = 0)', 
            axes=['y', 'z'], # Y and Z axes
            axes_limits=[axes_limits[1], axes_limits[2]],
            point_eliminations=point_eliminations
        )
    if data.shape[1] == 4: 
        fig.colorbar(ax3[2].collections[0], ax=ax3[2])

    fig.suptitle('Projections of LiDAR pointcloud data', fontsize=18)
    if savefig: 
        if filename: 
            plot_dir = os.path.dirname(os.path.abspath(__file__))
            plot_dir = os.path.join(plot_dir, "../../plots/") + str(filename) + '.png'
            plt.savefig(plot_dir)  
        else: 
            print("Did not save figure. Add filename in arguments to be able to save figure.")
    plt.show()

    os.path.join(os.path.dirname(__file__), '..')