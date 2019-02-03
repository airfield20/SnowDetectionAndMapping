#!/usr/bin/env python
import roslib
from scipy.spatial import distance as dist
import numpy as np
import cv2
import cv2.aruco as aruco
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def order_points(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def slope(p1, p2):
    num = (p2[1] - p1[1])
    den = (p2[0] - p1[0])
    if den != 0.:
        return num / den
    else:
        return num / 0.001


def get_point_on_line(x1, y1, m, x):
    y = m * (x - x1) + y1
    return [x, y]

def get_point_on_line_y(x1, y1, m, y):
    x = ((y-y1)/m) + x1
    return [x, y]


def expand_points_horizontally(pts, w):  # horizontally is x axis
    #rect = order_points(pts)
    (tl, tr, br, bl) = pts
    # expand horizontally
    ntr = get_point_on_line(tr[0], tr[1], slope(tl, tr), tr[0] + w / 2)
    ntl = get_point_on_line(tl[0], tl[1], slope(tl, tr), tl[0] - w / 2)
    nbr = get_point_on_line(br[0], br[1], slope(bl, br), br[0] + w / 2)
    nbl = get_point_on_line(bl[0], bl[1], slope(bl, br), bl[0] - w / 2)
    return np.array([ntl, ntr, nbr, nbl], dtype="float32")


def expand_points_vertically(pts, h):  # vertically is y axis
    # rect = order_points(pts)
    (tl, tr, br, bl) = pts
    ntr = get_point_on_line_y(tr[0], tr[1], slope(tr, br), tr[1] - h / 2)
    ntl = get_point_on_line_y(tl[0], tl[1], slope(tl, bl), tl[1] - h / 2)
    nbr = get_point_on_line_y(br[0], br[1], slope(br, tr), br[1] + h / 2)
    nbl = get_point_on_line_y(bl[0], bl[1], slope(bl, tl), bl[1] + h / 2)
    return np.array([ntl, ntr, nbr, nbl], dtype="float32")


def show_highlighted_image(image, rect, new_rect):
    (tl, tr, br, bl) = new_rect
    (otl, otr, obr, obl) = rect
    cv2.line(image, (tl[0], tl[1]), (tr[0], tr[1]), (0, 255, 0), 1)
    cv2.line(image, (bl[0], bl[1]), (br[0], br[1]), (0, 255, 0), 1)
    cv2.line(image, (tl[0], tl[1]), (bl[0], bl[1]), (0, 255, 0), 1)
    cv2.line(image, (br[0], br[1]), (tr[0], tr[1]), (0, 255, 0), 1)
    cv2.line(image, (otl[0], otl[1]), (otr[0], otr[1]), (0, 0, 255), 1)
    cv2.line(image, (obl[0], obl[1]), (obr[0], obr[1]), (0, 0, 255), 1)
    cv2.line(image, (otl[0], otl[1]), (obl[0], obl[1]), (0, 0, 255), 1)
    cv2.line(image, (obr[0], obr[1]), (otr[0], otr[1]), (0, 0, 255), 1)
    cv2.imshow('drawnImage', image)
    cv2.waitKey(1)

def four_point_transform(image, pts):
    rect = order_points(pts)
    expansion_width = 150
    expansion_height = 200
    new_rect = expand_points_horizontally(rect,expansion_width)
    new_rect = expand_points_vertically(new_rect,expansion_height)
    (tl, tr, br, bl) = new_rect
    rect = new_rect
    # (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


class image_transform:
    def __init__(self):
        self.image_pub = rospy.Publisher("/left/transformed_image", Image, queue_size=2)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/left/image_rect_gray", Image, self.callback)
        # self.pts = np.array([[505,381],[755,381],[1194,625],[182,625]])  # tl, tr, br, bl
        # self.pts = np.array([[414, 450], [878, 450], [1194, 625], [182, 625]])  # tl, tr, br, bl
        self.pts = np.array([[217, 135], [436, 135], [552, 301], [69, 301]])  # tl, tr, br, bl
        self.calibrated = False
        self.autoCalibrate = True

    def calibrate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        try:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            print("Found aruco tag with these corners: " + str(corners[0][0]))
            self.pts = corners[0][0]
            self.calibrated = True
        except:
            self.calibrated = False

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.calibrated:
                warped = four_point_transform(cv_image, self.pts)
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(warped, "bgr8"))
                except CvBridgeError as e:
                    print(e)
            elif self.autoCalibrate:
                self.calibrate(cv_image)

            else:
                self.calibrated = True
        except CvBridgeError as e:
            print(e)



def main():
    rospy.init_node('image_transformer', anonymous=False)
    it = image_transform()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
