import os
import numpy as np
import matplotlib.pyplot as plt
from src.waypoint_list import WayPoints


file_path = './acceleration.log'


with open(file_path, 'r') as file:  #reads data
    log_raw = file.read()


log_processed = []

for line in log_raw.splitlines():  #isolates the relavant numbers from the rest of the line
    log_processed.append(line[34:-1])


x = []
y = []
accel = []

for data in log_processed:          # splits data
    datas = data.split()
    x.append(datas[0])
    y.append(datas[1])
    accel.append(datas[2])



time = np.arange(len(accel))
np_accel = np.array(accel, dtype = "float")
np_x = np.array(x, dtype = "float")
np_y = np.array(y, dtype = "float")

waypoints = WayPoints()
pos_list = waypoints.getWayPoints()

x_list = np.array([point[0] for point in pos_list])
y_list = np.array([point[1] for point in pos_list])


plt.plot(time, np_accel, ".")
plt.title("Acceleration vs Time")
plt.xlabel("Time(s)")
plt.ylabel("Acceleration (m/s^2)")
plt.ylim(-15,15)
plt.grid(True)

plt.show()

plt.plot(np_x, np_y, ".")#linestyle = "-")
plt.plot(x_list, y_list, ".")
plt.plot(x_list[0], y_list[0], ".")
plt.title("X vs Y trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis("equal")

plt.show()