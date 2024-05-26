#!/usr/bin/env python3

# Copyright (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
# OBGym is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import sys

ROOT_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.insert(0, ROOT_PATH)

import rospy
from centernetplus import CenterNetPlus
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from rgor_sys.msg import DetectionObj, MultiDetectionObj
import numpy as np


class CenterNetCB:
    def __init__(self, nms_thresh=0.2, k=50, intra_threads=4):
        self.bridge = CvBridge()
        self.detector = CenterNetPlus(
            os.path.join(ROOT_PATH, "weights/centernetplus_3.onnx"),
            nms_thresh=nms_thresh,
            k=k,
            intra_threads=intra_threads,
        )

        self.publisher = rospy.Publisher(
            "/rgor/object", MultiDetectionObj, queue_size=1
        )
        self.pca_components = np.load(os.path.join(ROOT_PATH, "weights/pca.npy"))
        self.pca_mean = np.load(os.path.join(ROOT_PATH, "weights/mean.npy"))
        self.pca_variance = np.load(os.path.join(ROOT_PATH, "weights/variance.npy"))

    def __call__(self, data):
        color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        box, xywh, score, cls_ind, cls_vct, feat = self.detector(color_image)
        desc = (self.pca_components.dot(cls_vct.T).T - self.pca_mean) / np.sqrt(
            self.pca_variance
        )
        pdesc = desc.copy()
        pdesc[pdesc < 0] = 0
        ndesc = desc.copy()
        ndesc = -ndesc
        ndesc[ndesc < 0] = 0
        desc = np.concatenate((pdesc, ndesc), axis=1)
        desc = desc / np.max(desc, axis=1, keepdims=True)
        # desc = cls_vct
        ind = np.argmax(desc, axis=1)
        obj_list = []
        for i in range(box.shape[0]):
            obj = DetectionObj()
            obj.x = xywh[i][1]
            obj.y = xywh[i][0]
            obj.w = xywh[i][2]
            obj.h = xywh[i][3]
            obj.class_id = ind[i]
            obj.class_prob = desc[i].tolist()
            obj_list.append(obj)
        multi_obj = MultiDetectionObj()
        multi_obj.header = data.header
        multi_obj.obj = obj_list
        self.publisher.publish(multi_obj)


if __name__ == "__main__":
    rospy.init_node("centernet", anonymous=False)
    nms_thresh = rospy.get_param("~centernet/nms_thresh", 0.05)
    k = rospy.get_param("~centernet/k", 50)
    intra_threads = rospy.get_param("~centernet/intra_threads", 4)
    color_topic = rospy.get_param("~ros/rgb_topic", "/rgor/camera/color")

    centernet_cb = CenterNetCB(nms_thresh=nms_thresh, k=k, intra_threads=intra_threads)
    print("CenterNetPlus is ready.")
    color = rospy.Subscriber(color_topic, Image, centernet_cb, queue_size=1)
    rospy.spin()
