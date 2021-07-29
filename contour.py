
import cv2
import numpy as np
from scipy.spatial import distance


########################################################################################################################

def bounding_box(c):
    min_x = min([p[0] for p in c])
    min_y = min([p[1] for p in c])
    max_x = max([p[0] for p in c])
    max_y = max([p[1] for p in c])

    return [min_x, min_y, max_x, max_y]

########################################################################################################################

def compute_distance_transform_for_contour(c, image):

    for index in range(len(c)-1):
        cv2.line(image, tuple(np.array(np.round(c[index]), dtype=np.int32)), tuple(np.array(np.round(c[index+1]), dtype=np.int32)), 0)

    dist_trans = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    return dist_trans

########################################################################################################################

def compute_contour_distance(dist_trans, t_c2):
    total_dist = 0.0
    for p in t_c2:
        total_dist += dist_trans[int(round(p[1])), int(round(p[0]))]

    norm_distance = total_dist / len(t_c2)
    return norm_distance


########################################################################################################################


def contour_length(c):
    contour_length = 0.0
    for index in range(len(c) - 1):
        contour_length += distance.euclidean(c[index], c[index+1])

    return contour_length

########################################################################################################################

def contour_distance(c1, c2, max_countour_dist):

    if len(c1) == 0 or len(c2) == 0:
        return 0
    # create a distance transform out of c1:
    c_bb = bounding_box(c1)
    c_bb_width = c_bb[2] - c_bb[0]
    c_bb_height = c_bb[3] - c_bb[1]
    bb_plus_margin = [c_bb[0] - max_countour_dist, c_bb[1] - max_countour_dist,
                      c_bb[2] + max_countour_dist, c_bb[3] + max_countour_dist]

    # create an image in the size of the bb_plus_margin
    max_countour_dist = int(max_countour_dist)
    image = np.ones((c_bb_height + max_countour_dist * 2, c_bb_width + max_countour_dist * 2), dtype=np.uint8 ) * 255
    t_x = -c_bb[0] + max_countour_dist
    t_y = -c_bb[1] + max_countour_dist
    t_c1 = [(p[0] + t_x, p[1] + t_y) for p in c1]
    t_c2 = [(p[0] + t_x, p[1] + t_y) for p in c2]

    dist_trans = compute_distance_transform_for_contour(t_c1, image)

    average_error_per_sampled_point = compute_contour_distance(dist_trans, t_c2)

    return average_error_per_sampled_point


########################################################################################################################

if __name__ == "__main__":
    # find the distance between 2 input contours
    c1 = [(2, 5), (3 ,3), (5 ,2), (7, 3), (9 ,5)]
    c2 = [(1, 3), (3 ,3), (5 ,2), (7, 3), (9 ,5)]
    max_contour_dist = 10
    cd = contour_distance(c1, c2, max_contour_dist)
    print (cd)
