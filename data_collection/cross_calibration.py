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
        ts = ApproximateTimeSynchronizer([image_sub, pcl_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.callback)

    def callback(self, image, cloud):
        print(image.header.stamp)
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        print("Camera", image.header.stamp, image.header.seq)
        points_list = []

        for p in point_cloud2.read_points(cloud, skip_nans=True):
            points_list.append([p[0],p[1], p[2], p[3]])

        pcl_data = np.array(points_list) # XYZI
        print("LIDAR", cloud.header.stamp, pcl_data.shape)

if __name__ == '__main__':
    rospy.init_node("cross_calibration_node", anonymous=True)
    my_node = CrossCalibrationNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
