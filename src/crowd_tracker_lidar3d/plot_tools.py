import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colorbar

def plot_pointcloud3d(data): 
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], c=data['intensity'], s=50, edgecolor='', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.title('3D Point Cloud data')
    plt.show()
    
def draw_point_cloud(data, ax, title, axes=['x', 'y', 'z'], xlim3d=None, ylim3d=None, zlim3d=None):
        no_points = data.shape[0]
        point_size = 10**(3- int(np.log10(no_points))) # Adjust point size based on point cloud size
        
        if data.shape[1] == 4: # If point cloud is XYZI format (I = intensity)
            im = ax.scatter(*np.transpose(data[axes].to_numpy()), s = point_size, c=data['intensity'], cmap='viridis')
        elif data.shape[1] == 3:   # If point cloud is XYZ format 
            im = ax.scatter(*np.transpose(data[axes].to_numpy()), s = point_size, c='b', alpha = 0.7)
        
        ax.set_xlabel('{} axis'.format(axes[0]), fontsize=16)
        ax.set_ylabel('{} axis'.format(axes[1]), fontsize=16)
        
        x_max, x_min = np.max(data.x), np.min(data.x)
        y_max, y_min = np.max(data.y), np.min(data.y)
        z_max, z_min = np.max(data.z), np.min(data.z)
        intensity_max, intensity_min = np.max(data.intensity), np.min(data.intensity)
        
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