import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import rospy
from sensor_pkg.msg import *

step = 6
base_mat_x = []
for i in range(5):
    base_mat_x.append(step*(i+1))
base_mat_y = [0, 0, 0, 0, 0]

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=5):
        self.numpoints = numpoints
        self.stream = self.data_stream()
        self.curr_xyz = np.zeros((self.numpoints,3))

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s = next(self.stream).T
        self.scat_base = self.ax.scatter(base_mat_x, base_mat_y, c='grey', s=1500, alpha=0.4)
        self.scat = self.ax.scatter(x, y, c='green', s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")

        self.ax.axis([0, max(base_mat_x)+step, -1.5, 1.5])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        while True:
            xyz = self.curr_xyz.copy()
            # print(xyz.shape)
            # print(xyz)
            s = np.sqrt(np.sum(xyz**2, axis=1))*1500
            # print(xyz)
            offsets = np.array(base_mat_x)
            offsets = np.flip(offsets)

            # print(s)
            xyz[:,0] = (step/2)*xyz[:,0] + offsets
            # xyz[:,0] = (step/2)*xyz[:,0] + np.array(base_mat_x)
            # print(s)
            yield np.c_[xyz[:,0], xyz[:,1], s] #

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        
        # print(data)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(data[:, 2])
        # Set colors..
        # self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def normalize(self, xyz):
        nomarlize_data = []
        for i in range(5):
            data= np.array(xyz[i])
            nomarlize_data.append((data - np.min(data)) / (np.max(data) - np.min(data)))
        return nomarlize_data
    
    # Callback function definition
    def callback(self, sensor_data):

        # Initialize a list of lone_sensor messages to store the data that will be read
        sensor_values = [lone_sensor() for i in range(sensor_data.length)]
        
        print('------******-----///////-----///////')
        for i, sensor in enumerate(sensor_data.data):
            if sensor.is_present == False:             # Check if the sensor is present
                sensor_values[i].id = None                          # if not : set id to None, sensor will be displayed as "Sensor None"
            else:
                sensor_values[i] = sensor                           # If sensor is present, then copy the informations in the lone_sensor message
            # Then print each sensor with its X Y Z coordinates
            # print("\t Sensor-ID: {} \n".format(sensor_values[i].id))
            print("\t fx:{}, fy:{}, fz:{} \n".format(sensor_values[i].fx, sensor_values[i].fy, sensor_values[i].fz))
        
        # process the data
        sensor_values_new = sensor_values
        xyz = []
        # print(len(sensor_values))
        for j in range(len(sensor_values_new)):
            # fx and fy: shrink by 1000 times, fz: shrink by 10000 times
            xyz.append([sensor_values_new[j].fx/1000, sensor_values_new[j].fy/1000, sensor_values_new[j].fz/10000])
        
        xyz1 =[ [0, 0.5, 0.5], [0, -0.5, 0.5], [-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 1] ]
        print("xyz: ", xyz1)
        print('------------')
        print("xyz: ", xyz)
        self.curr_xyz = np.array(xyz)


if __name__ == '__main__':
    a = AnimatedScatter(5)

    rospy.init_node('subscriber_sensor_topic', anonymous=True)  # Initialize a node

    # Subscribe to the AllSensors Topic, in which informations read about sensors are published
    rospy.Subscriber("R_AllSensors", AllSensors, a.callback)
    
    plt.show()

    # Keeps the node listening until you stop the script manually
    rospy.spin()
