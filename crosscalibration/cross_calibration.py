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


        # rospy
        self.bridge = CvBridge()
        self.cc_pub = rospy.Publisher('cc', Image, queue_size=10)
        self.image_sub = Subscriber('camera/image', Image)
        self.pcl_sub = Subscriber('/livox/lidar',  PointCloud2)
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.pcl_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        # cross calib params
        self.frame = 0
        self.limit = 2

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.corners_found = 0

    def callback(self, image, cloud):
        print(image.header.stamp, self.frame)
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            self.imgpoints.append(corners2)
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            self.corners_found += 1

        points_list = []

        for p in point_cloud2.read_points(cloud, skip_nans=True):
            points_list.append([p[0],p[1], p[2], p[3]])

        pcl_data = np.array(points_list) # XYZI
        # print("LIDAR", cloud.header.stamp, pcl_data.shape)
        
        self.frame += 1
        if self.frame == self.limit:
            self.image_sub.unregister()
            self.pcl_sub.unregister()
            print("Calibrating ...")
            # assert self.corners_found > 0, "Must find corners for camera calibration"
            # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

            # # transform the matrix and distortion coefficients to writable lists
            # data = {'camera_matrix': np.asarray(mtx).tolist(),
            #         'dist_coeff': np.asarray(dist).tolist()}
            # print(data)
            camera_matrix = np.array([[646.2402992060081, 0.0, 621.3318167165827], 
                                    [0.0, 645.3381333548518, 497.8438155757953], 
                                    [0.0, 0.0, 1.0]]).T
            pcl_points = pcl_data.copy()
            pcl_points[:, 3] = 1 # X Y Z 1
            pixel_size_x = 4 # real world units per pixel
            pixel_size_y = 4
            tr = np.array(([1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1], 
                            [0, 0, 0]))
            image_points = np.matmul(np.matmul(pcl_points, tr), camera_matrix)
            image_xy = image_points[:, :2]
            image_xy[:, 0] = image_xy[:, 0]// pixel_size_x
            image_xy[:, 1] = image_xy[:, 1]// pixel_size_y
            yxs = []
            image_width = 1280
            image_height = 960
            for x,y in image_xy:
                # print(x, y)
                if (0 < x < image_width) and (0 < y < image_height):
                    yxs.append([y, x])
            yxs = np.array(yxs, dtype=np.int32)
            cv_image[yxs[:, 0], yxs[:, 1]] = [0, 0, 0]
            self.cc_pub.publish(self.bridge.cv2_to_imgmsg(cv_image))

if __name__ == '__main__':
    rospy.init_node("cross_calibration_node", anonymous=True)
    my_node = CrossCalibrationNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
