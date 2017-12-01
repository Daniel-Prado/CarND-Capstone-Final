from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

import time

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, accel_limit, wheel_radius, fuel_capacity,
	wheel_base, steer_ratio, max_lat_accel, max_steer_angle,
	tau, ts,
	kp, ki, kd):

        # Initialize controller attributes
	self.vehicle_mass = vehicle_mass
	self.accel_limit = accel_limit
	self.wheel_radius = wheel_radius
	self.fuel_capacity = fuel_capacity

	# Initialize itilities
	self.pid = PID(kp, ki, kd)
	self.low_pass_filter = LowPassFilter(tau, ts)
	self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed = ONE_MPH, max_lat_accel, max_steer_angle)

	# Need time for throttle calculation?
	self.previous_time = None


    # Brake = Desired acceleration, weight of the vehicle, and wheel radius
    
    def control(self, current_velocity, twist_cmd):
        # If time = None, initialize
        if self.previous_time is None:
            self.previous_time =  time.time() 
        
        # Calculate throttle using PID
        # Throttle values should be in the range 0 to 1
        elapsed_time = time.time() - self.previous_time
        self.previous_time =  time.time()
        throttle = min(1.0, self.pid.step(error, elapsed_time)
        
        # Calculate brake
        # Brake values should be in units of torque (N*m)
        # TODO: check if accel_limit is the right param to pass
        #https://discussions.udacity.com/t/what-is-the-range-for-the-brake-in-the-dbw-node/412339
        brake = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.accel_limit * self.wheel_radius
        
        #TODO: use decel_limit, accel_limit
        
        # Calculate steer
        #Good explanation on what to pass to get_steer function in forum:
        #https://discussions.udacity.com/t/no-able-to-keep-the-lane-with-yaw-controller/433887/5
        linear_velocity = twist_cmd.twist.linear.x
        angular_velocity = twist_cmd.twist.angular.z
        current_vel = current_velocity.twist.linear.x
        
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_vel)
        
        return throttle, brake, steer
