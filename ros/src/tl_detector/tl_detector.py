#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
# TODO remove imports after data is collected
import time
import csv
import math

STATE_COUNT_THRESHOLD = 3


def pose_distance(pose1, pose2):
    return math.sqrt((pose1.position.x - pose2.position.x)**2 + (pose1.position.y - pose2.position.y)**2)


def pose_angle(pose1, pose2):
    return math.atan2(pose2.position.y - pose1.position.y, pose2.position.x - pose1.position.x)


class TLDetector(object):

    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        rospy.logwarn("config %s", self.config)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # rospy.spin()
        self.loop()

    def loop(self):
        rate = rospy.Rate(1)
        myfile = open('../../sdc-data/image-data.csv', 'a+')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        while not rospy.is_shutdown():
            # TODO: This is to collect data only; remove later
            if self.camera_image is None or not self.lights or not self.pose:
                continue

            image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            image_name = "../../sdc-data/image" + str(time.time()) + ".jpeg"
            cv2.imwrite(image_name, image)

            data = (image_name, self.gt_image_status())
            wr.writerow(data)

            rate.sleep()

    def gt_image_status(self):
        # rospy.logwarn("location %s", self.pose)
        can_see_light = []
        for light in self.lights:
            if self.can_i_see_this_light(light):
                can_see_light.append(light.state)

        assert len(can_see_light) < 2
        if not can_see_light:
            return TrafficLight.UNKNOWN
        return can_see_light[0]

    def can_i_see_this_light(self, light):
        # distance=[]
        # for light in self.lights:
        distance = pose_distance(self.pose.pose, light.pose.pose)
        if distance > 100 or distance < 10:
            return False
        theta_light_car = pose_angle(self.pose.pose, light.pose.pose)
        if abs(theta_light_car - self.pose.pose.orientation.z) < 0.3:
            return True

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        # rospy.logwarn("traffic %s", len(msg.lights))
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # TODO implement
        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        # TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
