#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import EntityState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import euler_to_quaternion, quaternion_to_euler
import time


class vehicleController():

    def __init__(self, node=None):
        self.node = Node('vehicle_controller')
        self.own_node = True
            
        self.controlPub = self.node.create_publisher(AckermannDrive, "/ackermann_cmd", 1)
        self.prev_vel = 0
        self.L = 1.75 
        self.log_acceleration = False

       


    # Tasks 1: Read the documentation https://docs.ros.org/en/ros2_packages/humble/api/simulation_interfaces/srv/GetEntityState.html 
    #       and https://docs.ros.org/en/ros2_packages/humble/api/gazebo_msgs/msg/EntityState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    
    #   currentPose: GetEntityState response, the current state of the vehicle
    #   currentPose.pose: Point position, Quaternion orientation; representation of pose in free space, composed of position and orientation. 
    #   currentPose.twist: Vector3 linear, Vector3 angular; expresses velocity in free space broken into its linear and angular parts.
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0

        currentState = currentPose.state
        pose = currentState.pose
        twist = currentState.twist

        # Get (x, y) from currentPose.pose.position
        pos_x = pose.position.x
        pos_y = pose.position.y

        # Get yaw from currentPose.pose.orientation
        quaternion_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] # [x, y, z, w]
        euler_orientation = quaternion_to_euler(quaternion_orientation) # [roll, pitch, yaw]
        yaw = euler_orientation[2]

        # Get velocity from currentPose.twist; assuming z velocity is 0
        vel = math.sqrt(twist.linear.x ** 2 + twist.linear.y ** 2)


        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel, yaw # note that yaw is in radians


    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity

    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################
        target_velocity = 10

        straight_velocity = 12
        curve_velocity = 8

        # Based on the current position, we need the following two waypoints
        # If the two waypoints are in a line with current position, continue straight
        # Otherwise, slow down

        waypoint1 = future_unreached_waypoints[0]
        waypoint2 = future_unreached_waypoints[1]
        
        delta_y = waypoint2[1] - waypoint1[1] 
        delta_x = waypoint2[0] - waypoint1[0]
        line_yaw = np.arctan2(delta_y, delta_x)

        yaw_diff = min(abs(line_yaw - curr_yaw), np.pi * 2 - abs(line_yaw - curr_yaw)) # shortest distance between two angles on a circle
        if yaw_diff < 0.35: # 20 degrees
            return straight_velocity

        ####################### TODO: Your TASK 2 code ends Here #######################
        return curve_velocity

        
        

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):
       
        ####################### TODO: Your TASK 3 code starts Here #######################
        target_steering = 0

        lookahead_point = [target_point[0], target_point[1]] # same as target_point [x, y]

        delta_y = lookahead_point[1] - curr_y
        delta_x = lookahead_point[0] - curr_x
        ld = math.sqrt(delta_y ** 2 + delta_x ** 2)

        line_yaw = np.arctan2(delta_y, delta_x)
        alpha = line_yaw - curr_yaw

        if alpha > np.pi:
            alpha -= 2 * np.pi
        elif alpha < -np.pi:
            alpha += 2 * np.pi

        target_steering = math.atan((2 * self.L * math.sin(alpha)) / ld)
        ####################### TODO: Your TASK 3 code starts Here #######################
        return target_steering
       





    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: GetEntityState response, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        
        if currentPose is None:
            print("Warning: No current pose data")
            return
            
        if len(future_unreached_waypoints) == 0:
            print("Warning: No waypoints available")
            return

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        if self.log_acceleration:
            acceleration = (curr_vel - self.prev_vel) * 100  # Since we are running at 100Hz

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)

        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = float(target_velocity)
        newAckermannCmd.steering_angle = float(target_steering)

        self.controlPub.publish(newAckermannCmd)
        
        # Store current velocity for next iteration
        self.prev_vel = curr_vel

    def stop(self):
        """Stop the vehicle by setting speed to 0 and steering to 0"""
        try:
            newAckermannCmd = AckermannDrive()
            newAckermannCmd.speed = 0.0
            newAckermannCmd.steering_angle = 0.0
            self.controlPub.publish(newAckermannCmd)
            print("Controller: Stop command sent")
        except Exception as e:
            print(f"Controller: Error sending stop command: {e}")
        
    def destroy(self):
        if self.own_node:
            self.node.destroy_node()