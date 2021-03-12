from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
import rospy
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from sensor_msgs import point_cloud2 

class CrossCalibrationNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        image_sub = Subscriber('camera/image', Image)
        pcl_sub = Subscriber('/livox/lidar',  PointCloud2)
        ts = ApproximateTimeSynchronizer([image_sub, pcl_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)
        image_sub.registerCallback(self.camera_callback)
        pcl_sub.registerCallback(self.lidar_callback)

    def camera_callback(self, data):
        # Solve all of perception here...
        print(data.header.stamp)
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        print("Camera", data.header.stamp, data.header.seq)

    def lidar_callback(self, data):
        # Solve all of perception here...
        points_list = []

        for p in point_cloud2.read_points(data, skip_nans=True):
            points_list.append([p[0],p[1], p[2], p[3]])

        pcl_data = np.array(points_list) # XYZI
        print("LIDAR", data.header.stamp, data.header.seq)

    def callback(self, image, pcl):
        # Solve all of perception here...
        print('')


if __name__ == '__main__':
    rospy.init_node("cross_calibration_node", anonymous=True)
    my_node = CrossCalibrationNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
