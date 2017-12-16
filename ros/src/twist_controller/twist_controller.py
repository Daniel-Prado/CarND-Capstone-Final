from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

import time
import rospy
from math import fabs

GAS_DENSITY_KG_GAL = 2.858
GAS_DENSITY_KG_CUB_M = 755.00373
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, accel_limit, decel_limit, wheel_radius, fuel_capacity,
            wheel_base, steer_ratio, max_lat_accel, max_steer_angle,
            tau, ts,
            kp, ki, kd):

        # Initialize controller attributes
        self.vehicle_mass = vehicle_mass
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.wheel_radius = wheel_radius
        self.fuel_capacity = fuel_capacity

        # Initialize itilities
        self.pid = PID(kp, ki, kd)
        self.low_pass_filter = LowPassFilter(tau, ts) #see if can pass real ts
	self.steer_lpf = LowPassFilter(tau=3, ts=1)
        self.yaw_controller = YawController(wheel_base, steer_ratio, ONE_MPH, max_lat_accel, max_steer_angle)

        # Need time for throttle calculation?
        self.previous_time = None

    # Brake = Desired acceleration, weight of the vehicle, and wheel radius
    def control(self, current_velocity, twist_cmd):
        # If time = None, initialize
        if self.previous_time is None:
            self.previous_time =  time.time() 
        
        linear_velocity = twist_cmd.twist.linear.x
        angular_velocity = twist_cmd.twist.angular.z
        current_vel = current_velocity.twist.linear.x
        error = linear_velocity - current_vel
        
        # Calculate throttle using PID
        # Throttle values should be in the range 0 to 1
        elapsed_time = time.time() - self.previous_time
        self.previous_time = time.time()
        throttle = self.pid.step(error, elapsed_time)
        throttle = min(self.accel_limit, throttle)
        
        # Calculate brake
        # Brake values should be in units of torque (N*m)
        #https://discussions.udacity.com/t/what-is-the-range-for-the-brake-in-the-dbw-node/412339
        brake = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY_KG_CUB_M) * throttle * self.wheel_radius
        brake = self.low_pass_filter.filt(brake)
        brake = max(self.decel_limit, brake)
        
        # The param contains speed limit in kmph
        # Let's keep it 3 km below speed limit to prevent speed violations
        speed_limit = ((rospy.get_param('/waypoint_loader/velocity') - 3)
                        * 1000.) / (60. * 60.)
        if current_vel >= speed_limit:
            throttle = 0.0
            brake = 0.0
        
        # Calculate steer
        #Good explanation on what to pass to get_steer function in forum:
        #https://discussions.udacity.com/t/no-able-to-keep-the-lane-with-yaw-controller/433887/5
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_vel)
	steer = self.steer_lpf.filt(steer)
        return throttle, brake, steer
