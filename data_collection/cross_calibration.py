import message_filters
from sensor_msgs.msg import Image, PointCloud2
import rospy
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from sensor_msgs import point_cloud2 
import pptk

class CrossCalibrationNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        # image_sub = rospy.Subscriber('camera/image', Image, self.camera_callback)
        pcl_sub = rospy.Subscriber('/livox/lidar',  PointCloud2, self.lidar_callback)
        # ts = message_filters.TimeSynchronizer([image_sub, pcl_sub], 10)
        # ts.registerCallback(self.callback)

    def camera_callback(self, data):
        # Solve all of perception here...
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        print(cv_image.shape)

    def lidar_callback(self, data):
        # Solve all of perception here...
        points_list = []

        for p in point_cloud2.read_points(data, skip_nans=True):
            points_list.append([p[0],p[1], p[2]])

        pcl_data = np.array(points_list)
        print(pcl_data.shape)
        pptk.viewer(pcl_data)
        print('od')

    def callback(self, image, pcl):
        # Solve all of perception here...
        print(image, pcl)


if __name__ == '__main__':
    my_node = CrossCalibrationNode()
    rospy.init_node("cross_calibration_node", anonymous=True)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
