#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):

	rospy.logwarn("Inside Waypoint Updater")

        rospy.init_node('waypoint_updater')

	self.current_pose = None
        self.base_waypoints = None
        self.final_waypoints = []
        #self.traffic_waypoint = None
        #self.obstacle_waypoint = None

        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
	# rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
	# rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=LOOKAHEAD_WPS)

        # TODO: Add other member variables you need below

        rospy.spin()

    def current_pose_cb(self, msg):
	#rospy.logwarn("Update current_pose")
        self.current_pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
	total_waypoints=np.shape(waypoints.waypoints)[0]
	rospy.logwarn("Total waypoints: {}".format(total_waypoints))
	
	# If total number of waypoints is less than
	# LOOKAHEAD_WPS, send all waypoints
	for i in range(min(total_waypoints, LOOKAHEAD_WPS)):
		waypoint=waypoints.waypoints[i]
		rospy.logwarn("sample x: {}".format(waypoint.twist.twist.linear.x))
		self.final_waypoints.append(waypoint)

	rospy.logwarn("waypoints size: {}".format(len(self.final_waypoints)))

	#Publishing the Lane with final enpoints after taking first 200
	lane=Lane()
	lane.header=self.base_waypoints.header
	lane.waypoints=np.asarray(self.final_waypoints)
	self.final_waypoints_pub.publish(lane)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
