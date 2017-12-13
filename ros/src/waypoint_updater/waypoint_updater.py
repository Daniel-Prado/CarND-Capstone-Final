#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import distance

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.logwarn("Inside Waypoint Updater")
        
        rospy.init_node('waypoint_updater')
        
        self.current_pose = None
        self.base_waypoints = None
        self.final_waypoints = []
        #self.traffic_waypoint = None
        #self.obstacle_waypoint = None
        self.total_waypoints = 0
        self.last_closest_point = None
        
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=LOOKAHEAD_WPS)
        
        # TODO: Add other member variables you need below
        rospy.spin()

    def current_pose_cb(self, msg):
        self.current_pose = msg
        if self.base_waypoints is not None:
            #rospy.logwarn("Publishing from Waypoints Updater:")
    
            closest_point = self.find_closest_waypoint()
            rospy.logwarn("CLOSEST POINT {}".format(closest_point))

            self.final_waypoints = [] #Reinitialize each time
            for i in range(closest_point, closest_point+LOOKAHEAD_WPS):
                if i >= self.total_waypoints:
                    i = i - self.total_waypoints
                waypoint=self.base_waypoints.waypoints[i]
                self.final_waypoints.append(waypoint)
            #rospy.logwarn("waypoints size: {}".format(len(self.final_waypoints)))
            self.publish()

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.total_waypoints=np.shape(waypoints.waypoints)[0]
        rospy.logwarn("Total waypoints: {}".format(self.total_waypoints))

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def publish(self):
        #Publishing the Lane with final enpoints
        lane=Lane()
        lane.header=self.base_waypoints.header
        lane.waypoints=np.asarray(self.final_waypoints)
        self.final_waypoints_pub.publish(lane)

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance_between_waypoints(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def find_closest_waypoint(self):
        # Find one waypoint closest to current position of the car
        closest_point = 0
        closest_dist_so_far = 100000 #replace with highest float
        current_w_pos = self.current_pose.pose.position
        if self.last_closest_point is None:
            wp_search_list = list(range(0, self.total_waypoints))
        else:
            min_point = self.last_closest_point #assumes only forward movement
            max_point = (self.last_closest_point + 20) % self.total_waypoints
            if max_point > min_point:
                wp_search_list = list(range(min_point,max_point))
            else:
                wp_search_list = list(range(min_point, self.total_waypoints))
                #wp_search_list.append(list(range(0,max_point)))

        for i in wp_search_list:
            another_w_pos=self.base_waypoints.waypoints[i].pose.pose.position
            a = (current_w_pos.x, current_w_pos.y, current_w_pos.z)
            b = (another_w_pos.x, another_w_pos.y, another_w_pos.z)
            distance_between_wps = distance.euclidean(a, b)
            if(distance_between_wps<closest_dist_so_far):
                closest_dist_so_far=distance_between_wps
                closest_point = i
        self.last_closest_point = closest_point
        return closest_point

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
