#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from fire_utils import (
    transmission_homogeneous_medium,
    get_dark_channel,
    estimate_atmospheric_light_rf,
    haze_linear,
)



class FireSim:
    def __init__(self):
        self.alpha = 0
        self.beta = 1
        self.shift = [0, 0, 0]

        self.last_alpha = 0
        self.last_beta = 1
        self.last_shift = [0, 0, 0]

        self.time = 30
        self.t = 0

        self.random_args()
        self.bridge = CvBridge()

        self.img_pub = rospy.Publisher("/rgor/fire/color", Image, queue_size=10)
        self.depth_pub = rospy.Publisher("/rgor/fire/depth", Image, queue_size=10)

    def random_args(self):
        self.alpha = np.random.randint(-50, -10)
        self.beta = np.random.uniform(0.6, 1)
        self.alpha = -20
        self.beta = 0.8
        self.shift[0] = np.random.randint(10, 20)
        self.shift[1] = np.random.randint(-15, -5)
        self.shift[2] = np.random.randint(-20, -5)

    def cb(self, color_msg, depth_msg):
        color_image = (
            self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8") / 255.0
        )
        depth_image = (
            self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1") / 1000
        )  # 根据实际情况调整

        depth_image[depth_image == 0] = 20.0
        # t = transmission_homogeneous_medium(depth_image,  0.23, 462.266357421875,320,240)
        t = transmission_homogeneous_medium(
            depth_image, 0.1, 462.266357421875, 320, 240
        )

        clear_image_dark_channel = get_dark_channel(color_image, 15)

        L_atm = estimate_atmospheric_light_rf(clear_image_dark_channel, color_image)
        I = haze_linear(color_image, t, L_atm) * 255

        I = I.astype("uint8")

        if self.t == self.time:
            self.last_alpha = self.alpha
            self.last_beta = self.beta
            self.last_shift = self.shift
            self.random_args()
            self.t = 0
        else:
            self.t += 1

        alpha = self.last_alpha + (self.alpha - self.last_alpha) * self.t / self.time
        beta = self.last_beta + (self.beta - self.last_beta) * self.t / self.time
        shift = [
            self.last_shift[0]
            + (self.shift[0] - self.last_shift[0]) * self.t / self.time,
            self.last_shift[1]
            + (self.shift[1] - self.last_shift[1]) * self.t / self.time,
            self.last_shift[2]
            + (self.shift[2] - self.last_shift[2]) * self.t / self.time,
        ]

        I = cv2.convertScaleAbs(I, alpha, beta)

        # I[:, :, 2] = cv2.add(I[:, :, 2], shift[0])
        # I[:, :, 1] = cv2.add(I[:, :, 1], shift[1])
        # I[:, :, 0] = cv2.add(I[:, :, 0], shift[2])
        # I[:, :, 2] = np.clip(I[:, :, 2], 0, 255)
        # I[:, :, 1] = np.clip(I[:, :, 1], 0, 255)
        # I[:, :, 0] = np.clip(I[:, :, 0], 0, 255)

        self.img_pub.publish(
            self.bridge.cv2_to_imgmsg(I, encoding="bgr8", header=color_msg.header)
        )
        self.depth_pub.publish(depth_msg)



fire_sim = FireSim()


def listener():
    rospy.init_node("image_subscriber", anonymous=True)
    color_sub = message_filters.Subscriber("/rgor/camera/color", Image)
    depth_sub = message_filters.Subscriber("/rgor/camera/depth", Image)
    ts = message_filters.ApproximateTimeSynchronizer(
        [color_sub, depth_sub], 10, 0.1, allow_headerless=True
    )
    ts.registerCallback(fire_sim.cb)
    rospy.spin()


if __name__ == "__main__":
    listener()
