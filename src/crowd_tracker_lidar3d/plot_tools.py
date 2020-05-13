import numpy as np
import os 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


def plot_pointcloud3d(data): 
    no_points = data.shape[0]
    point_size = 10**(3- int(np.log10(no_points))) # Adjust point size based on point cloud size

    plt.ion()
    fig = plt.figure(figsize=[13,9])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], c=data['intensity'], s=point_size*2, edgecolor='', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.title('3D Point Cloud data')
    plt.show()
    
def draw_point_cloud(data, ax, title, axes=['x', 'y', 'z'], xlim3d=None, ylim3d=None, zlim3d=None, point_eliminations=None):
        no_points = data.shape[0]
        point_size = 10**(3- int(np.log10(no_points))) # Adjust point size based on point cloud size
        
        if data.shape[1] == 4: # If point cloud is XYZI format (I = intensity)
            im = ax.scatter(*np.transpose(data[axes].to_numpy()), s = point_size, c=data['intensity'], cmap='viridis')
            if point_eliminations is not None: 
                ax.scatter(*np.transpose(point_eliminations[axes].to_numpy()), s = point_size, c='r', alpha = 0.7)

        elif data.shape[1] == 3:   # If point cloud is XYZ format 
            im = ax.scatter(*np.transpose(data[axes].to_numpy()), s = point_size, c='b', alpha = 0.7)
        
        ax.set_xlabel('{} axis'.format(axes[0]), fontsize=16)
        ax.set_ylabel('{} axis'.format(axes[1]), fontsize=16)
        
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

def show_projections(raw_data, dimensions, savefig=False, filename=None, point_eliminations=None): 
    """
    [summary]

    Args:
        raw_data (pandas DataFrame): the raw data loaded from rosbags to a dataframe
        dimensions (list): containing the column names (string) of the dataframe 
                           (namely x,y,z), if 'intensity' is included, points are colored 
                           according to their intensity value
        savefig (bool, default=False)
    """
    
    data = raw_data[dimensions]
    if point_eliminations is not None:
        point_eliminations = point_eliminations[dimensions]
    fig, ax3 = plt.subplots(1, 3, figsize=(30, 15)) # plots in 3 columns
    # f, ax3 = plt.subplots(3, 1, figsize=(12, 25)) if plots in 1 column 

    im1 = draw_point_cloud(data,
            ax3[0], 
            'XZ projection (Y = 0)', 
            axes=['x', 'z'], # X and Z axes,
            point_eliminations=point_eliminations
        )

    im2 = draw_point_cloud(data,
            ax3[1], 
            'XY projection (Z = 0)', 
            axes=['x', 'y'], # X and Y axes
            point_eliminations=point_eliminations
        )

    im3 = draw_point_cloud(data,
            ax3[2], 
            'YZ projection (X = 0)', 
            axes=['y', 'z'], # Y and Z axes
            point_eliminations=point_eliminations
        )
    if len(dimensions) == 4: 
        fig.colorbar(ax3[2].collections[0], ax=ax3[2])

    fig.suptitle('Projections of LiDAR pointcloud data', fontsize=18)
    if savefig: 
        if filename: 
            plot_dir = os.path.dirname(os.path.abspath(__file__))
            plot_dir = os.path.join(plot_dir, "../../plots/") + str(filename) + '.svg'
            plt.savefig(plot_dir)  
        else: 
            print("Did not save figure. Add filename in arguments to be able to save figure.")
    plt.show()

    os.path.join(os.path.dirname(__file__), '..')