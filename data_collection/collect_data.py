from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
import rospy
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from sensor_msgs import point_cloud2 
import argparse
import sys

class DataCollectionNode(object):
    def __init__(self, location, max_count):
        self.location = location
        self.max_count = max_count
        self.bridge = CvBridge()
        image_sub = Subscriber('camera/image', Image)
        pcl_sub = Subscriber('/livox/lidar',  PointCloud2)
        ts = ApproximateTimeSynchronizer([image_sub, pcl_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.callback)
        self.save_dir = os.path.join('./data', self.location)
        os.makedirs(self.save_dir)
        self.count = 0
        print("Initialized data collection")

    def callback(self, image, cloud):
        stamp = image.header.stamp
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        # print("Camera", image.header.stamp, image.header.seq)
        points_list = []

        for p in point_cloud2.read_points(cloud, skip_nans=True):
            points_list.append([p[0],p[1], p[2], p[3]])

        pcl_data = np.array(points_list) # XYZI

        # save
        cv2.imwrite(os.path.join(self.save_dir, '{:09d}_{}.png'.format(self.count, stamp)), cv_image)
        np.save(os.path.join(self.save_dir, '{:09d}_{}.npy'.format(self.count, stamp)), pcl_data)

        # increment count
        self.count += 1
        print("{count:09d}_{stamp} -> {count}/{max_count}".format(count=self.count, stamp=stamp, max_count=self.max_count))
        if self.count == self.max_count:
            sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data-collection tool")
    parser.add_argument("-location", type=str, required=True, help='name of city in which data is being collected')
    parser.add_argument('-max_count', action="store", type=int, default=37000, help="max frames to collect")

    args = parser.parse_args()
    rospy.init_node("data_collection_node", anonymous=True)
    my_node = DataCollectionNode(args.location, args.max_count)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down data collection"
