from functools import partial
from matplotlib import pyplot as plt
from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.builder import rescale_images_to_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from pathlib import Path
import imutils
# import joblib
# import menpo.feature
import menpo.image
# import menpo.io as mio
import numpy as np
import cv2
import math
# import tensorflow as tf
# import detect
# import utils

# FIRMWARE PARAMS:

# INVERSE OF FIRMWARE PARAMS:
# CCV_R2Y = 0.2986
# CCV_G2Y = 0.5875
# CCV_B2Y = 0.1139
# CCV_b2Y = -0.1517
# CCV_R2U = -0.1684
# CCV_G2U = -0.3313
# CCV_B2U = 0.4997
# CCV_b2U = 126.9710
# CCV_R2V = 0.5015
# CCV_G2V = -0.4201
# CCV_B2V = -0.0814
# CCV_b2V = 127.4269

# STANDARD YUV2RGB:
CCV_Y2R = 1.0
CCV_U2R = 0.0
CCV_V2R = 1.4
CCV_b2R = -128.0 * 1.4
CCV_Y2G = 1.0
CCV_U2G = -0.343
CCV_V2G = -0.711
CCV_b2G = 128.0 * (0.343 + 0.711)
CCV_Y2B = 1.0
CCV_U2B = 1.765
CCV_V2B = 0.0
CCV_b2B = -128.0 * 1.765

# STANDARD RGB2YUV:
CCV_R2Y = 0.299
CCV_G2Y = 0.587
CCV_B2Y = 0.114
CCV_b2Y = 0.0
CCV_R2U = -0.169
CCV_G2U = -0.331
CCV_B2U = 0.5
CCV_b2U = 128.0
CCV_R2V = 0.5
CCV_G2V = -0.419
CCV_B2V = -0.081
CCV_b2V = 128.0

# gil
# TARGET_SIZE = 128
# TARGET_SIZE = 142
# TARGET_SIZE = 256
# TARGET_SIZE = 36
MARGIN = 0.3
MARGIN_X = 0.3
MARGIN_DOWN = 0.15
MARGIN_UP = 0.05
MARGIN_VEC = [MARGIN_X, MARGIN_X, MARGIN_UP, MARGIN_DOWN]
from common_params import get_bb_size
from common_params import get_initial_points_bb_size

BB_SIZE = get_bb_size()
INITIAL_POINTS_BB_SIZE = get_initial_points_bb_size()

def preprocess_image_y_only(image, bbs, target = BB_SIZE, margin = MARGIN):

    # convert the input rgb image to grayscale
    y_im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # get crop src and dst indexes
    h_im, w_im = y_im.shape
    bbs_m, bbs_n, bbs_n_x2_pad, bbs_n_y2_pad, w_larger = get_crop_indexes(bbs, w_im, h_im, target, margin)

    # resize
    y_crop = y_im[bbs_m[1]:bbs_m[3] + 1, bbs_m[0]:bbs_m[2] + 1].copy()
    y_resized = resize_bilinear(y_crop, bbs_n, False)
    # rotate
    y_rotated = y_resized

    # padding
    y_padded = np.zeros([bbs_n[3] + bbs_n_y2_pad, bbs_n[2] + bbs_n_x2_pad, 3], dtype = type(y_rotated))
    y_padded[bbs_n[1]:bbs_n[3] + 1, bbs_n[0]:bbs_n[2] + 1, :] = y_rotated

    # return
    return y_padded, bbs_m, bbs_n


########################################################################################################################



def compute_rect_intersection(rect, gt_rect):
    r_l = rect[0]
    r_t = rect[1]
    r_r = rect[2]
    r_b = rect[3]

    g_l = gt_rect[0]
    g_t = gt_rect[1]
    g_r = gt_rect[2]
    g_b = gt_rect[3]

    left = max(r_l, g_l)
    top = max(r_t, g_t)
    right = min(r_r, g_r)
    bottom = min(r_b, g_b)

    intersection_area = 0
    if left < right and bottom > top:
        intersection_area = (right - left) * (bottom - top)

    return intersection_area, [left, top, right, bottom]


########################################################################################################################

def crop_bbs_from_image(image, bbs, margin, grayscale=False):

    bbs_w = bbs[2] - bbs[0] + 1
    bbs_h = bbs[3] - bbs[1] + 1


    bbs_plus_margin_relative_to_input_image = np.array(my_round(np.array([bbs[0] - margin * bbs_w,
                                     bbs[1] - margin * bbs_h,
                                     bbs[2] + margin * bbs_w,
                                     bbs[3] + margin * bbs_h])), dtype=np.int32)


    input_plus_margin_w = bbs_plus_margin_relative_to_input_image[2] - bbs_plus_margin_relative_to_input_image[0] + 1
    input_plus_margin_h = bbs_plus_margin_relative_to_input_image[3] - bbs_plus_margin_relative_to_input_image[1] + 1

    if grayscale:
        output_image = np.zeros([input_plus_margin_h, input_plus_margin_w])
    else:
        output_image = np.zeros([input_plus_margin_h, input_plus_margin_w, 3])

    tx = -bbs_plus_margin_relative_to_input_image[0]
    ty = -bbs_plus_margin_relative_to_input_image[1]

    Tr = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])

    # gils - we subtract 1 here because our rectangle representation format. The lower right point is included
    input_image_bbs = [0, 0, image.shape[1] - 1 , image.shape[0] - 1]
    intersection_area, bbs_intersection =  compute_rect_intersection(bbs_plus_margin_relative_to_input_image, input_image_bbs)

    bbs_intersection = my_round(np.array(bbs_intersection))

    bbs_intersection_relative_to_cropped_image = [bbs_intersection[0] - bbs_plus_margin_relative_to_input_image[0],
                                                  bbs_intersection[1] - bbs_plus_margin_relative_to_input_image[1],
                                                  bbs_intersection[2] - bbs_plus_margin_relative_to_input_image[0],
                                                  bbs_intersection[3] - bbs_plus_margin_relative_to_input_image[1]]

    # Verify the dimesions match
    # w0 = bbs_intersection_relative_to_cropped_image[2] - bbs_intersection_relative_to_cropped_image[0] + 1
    # h0 = bbs_intersection_relative_to_cropped_image[3] - bbs_intersection_relative_to_cropped_image[1] + 1
    # w1 = bbs_intersection[2] - bbs_intersection[0] + 1
    # h1 = bbs_intersection[3] - bbs_intersection[1] + 1
    #
    # if w0 > w1:
    #     d = w0 - w1
    #     bbs_intersection_relative_to_cropped_image[2] -= d
    # elif w0 < w1:
    #     d = w1 - w0
    #     bbs_intersection[2] -= d
    # if h0 > h1:
    #     d = h0 - h1
    #     bbs_intersection_relative_to_cropped_image[3] -= d
    # elif h0 < h1:
    #     d = h1 - h0
    #     bbs_intersection[3] -= d

    if grayscale:
        output_image[bbs_intersection_relative_to_cropped_image[1]: bbs_intersection_relative_to_cropped_image[3],
        bbs_intersection_relative_to_cropped_image[0]: bbs_intersection_relative_to_cropped_image[2]] = \
            image[bbs_intersection[1]:bbs_intersection[3],
            bbs_intersection[0]:bbs_intersection[2]]

    else:

        output_image[bbs_intersection_relative_to_cropped_image[1]: bbs_intersection_relative_to_cropped_image[3]+1,
        bbs_intersection_relative_to_cropped_image[0]: bbs_intersection_relative_to_cropped_image[2]+1,:] = \
            image[bbs_intersection[1]:bbs_intersection[3]+1,
            bbs_intersection[0]:bbs_intersection[2]+1,:]



    bbs_relative_to_cropped_image = [bbs[0] - bbs_plus_margin_relative_to_input_image[0],
                                                  bbs[1] - bbs_plus_margin_relative_to_input_image[1],
                                                  bbs[2] - bbs_plus_margin_relative_to_input_image[0],
                                                  bbs[3] - bbs_plus_margin_relative_to_input_image[1]]


    # debug_image = np.uint8(output_image).copy()
    # cv2.rectangle(debug_image, (int(bbs_relative_to_cropped_image[0]), int(bbs_relative_to_cropped_image[1])),
    #               (int(bbs_relative_to_cropped_image[2]), int(bbs_relative_to_cropped_image[3])), (0, 255, 0), 3)
    #
    # cv2.imwrite('c:/temp/cropped_image.jpg', debug_image)
    return output_image, bbs_relative_to_cropped_image, Tr


########################################################################################################################

def correct_image_roll(croped_image_plus_margin, roll_angle, bbs_relative_to_cropped_image):
    cropped_image_height = croped_image_plus_margin.shape[0]
    cropped_image_width = croped_image_plus_margin.shape[1]
    center_x = (cropped_image_width) / 2.0
    center_y = (cropped_image_height) / 2.0

    M = cv2.getRotationMatrix2D((center_x, center_y), -roll_angle, 1.0)

    # We rotate the whole normalized image (with margins)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image

    nW = int((cropped_image_height * sin) + (cropped_image_width * cos))
    nH = int((cropped_image_height * cos) + (cropped_image_width * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - center_x
    M[1, 2] += (nH / 2) - center_y

    rotated_image = cv2.warpAffine(croped_image_plus_margin, M, (nW, nH))

    # Apply NFD logic to adjust the bounding box according to roll
    roll_rad = roll_angle * math.pi / 180.0

    w = float(bbs_relative_to_cropped_image[2] - bbs_relative_to_cropped_image[0] + 1)
    h = float(bbs_relative_to_cropped_image[3] - bbs_relative_to_cropped_image[1] + 1)

    # Aspect ratio transformation
    # w_tag / h_tag = pow(w / h, cos(2*roll))

    ar_out = (w / h) ** math.cos(2*roll_rad)

    # Area conversion:
    # w_tag * h_tag = w * h
    w_tag = math.sqrt(w * h * ar_out)
    h_tag = w * h / w_tag

    corrected_bb = np.array([(nW / 2) - w_tag / 2.0,
                    (nH / 2) - h_tag / 2.0,
                    (nW / 2) + w_tag / 2.0,
                    (nH / 2) + h_tag / 2.0], dtype=np.int32)
    M_homo = np.vstack([M, [0, 0, 1]])

    # debug_image = rotated_image.copy()
    # cv2.rectangle(debug_image, (corrected_bb[0], corrected_bb[1]), (corrected_bb[2], corrected_bb[3]), (0, 255, 0), 3)
    # cv2.imwrite('c:/temp/rotated_image.jpg', debug_image)
    return rotated_image, corrected_bb, M_homo




########################################################################################################################



def scale_image(roll_corrected_image_plus_margin, target, bbs_relative_to_roll_corrected_image):
    # Scale the input image so that the face bounding box larger dimension is set to target.
    # The other dimension is scaled to preserve the original aspect ration
    face_bb_width = bbs_relative_to_roll_corrected_image[2] - bbs_relative_to_roll_corrected_image[0] + 1
    face_bb_height = bbs_relative_to_roll_corrected_image[3] - bbs_relative_to_roll_corrected_image[1] + 1

    if face_bb_height > face_bb_width:
        scale = float(target) / float(face_bb_height)
    else:
        scale = float(target) / float(face_bb_width)

    scaled_image = cv2.resize(roll_corrected_image_plus_margin,dsize=(0,0), fx=scale, fy=scale)
    # REMOVE AFTER SCALE TEST
    # scaled_image = cv2.resize(roll_corrected_image_plus_margin, dsize=(0, 0), fx=scale/2, fy=scale/2)
    # scaled_image = cv2.resize(scaled_image, dsize=(0, 0), fx=2, fy=2)

    scaled_bb = np.array(my_round(bbs_relative_to_roll_corrected_image * scale ), dtype=np.int32)

    # debug_image = scaled_image.copy()
    # cv2.rectangle(debug_image, (scaled_bb[0], scaled_bb[1]), (scaled_bb[2], scaled_bb[3]), (0, 255, 0), 3)
    # cv2.imwrite('c:/temp/scaled_image.jpg', debug_image)

    Ts = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]])

    return scaled_image, scaled_bb, Ts

########################################################################################################################


def my_round(np_arr_input):
    """
    To align with SW implementation, we implement with a tie break that is different from the numpy implementation.
    numpy rounds to the nearest even integer so -0.5 rounds to 0 and 1.5 rounds to 2
    To keep things simpler we round x.5 to the nearest integer with the largest absolute value:
    0.5 rounds to 1, -2.5 rounds to -3
    :param np_array:
    :return:
    """

    sign = (np_arr_input >= 0)*2 - 1
    out = np.array(np_arr_input + sign*0.5, dtype=np.int32)
    return out

########################################################################################################################
def preprocess_image_with_roll(image, bbs, roll_angle, target= BB_SIZE, margin=MARGIN, grayscale=False):

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    bbs = my_round(np.array(bbs))
    # debug_image = image.copy()
    # cv2.rectangle(debug_image, (bbs[0], bbs[1]), (bbs[2], bbs[3]), (0, 255, 0), 3)
    # cv2.imwrite('c:/temp/input_image.jpg', debug_image)

    # First we add margin to the FD bbs and crop it from the input image
    # T0 transfer points from input image space to cropped image space
    croped_image_plus_margin, bbs_relative_to_cropped_image, T0 = crop_bbs_from_image(image, bbs, margin, grayscale)
    # plt.imsave('/home/gilsh/tmp/for_elad/dump/after_crop_and_pad.png', croped_image_plus_margin/255)
    # with open('/home/gilsh/tmp/for_elad/dump/after_crop_and_pad.txt', 'w') as f:
    #     np.save


    # now we rotate croped_image_plus_margin to cancel its roll angle
    # T1 transfer points from cropped image space to rotated image space
    roll_corrected_image_plus_margin, bbs_relative_to_roll_corrected_image, T1 = \
        correct_image_roll(croped_image_plus_margin, roll_angle, bbs_relative_to_cropped_image)

    # plt.imsave('/home/gilsh/tmp/for_elad/dump/after_roll.png', roll_corrected_image_plus_margin/255)

    # scale the roll corrected image such that its longer dimension is equal to the target size
    # T2
    scaled_roll_corrected_image_plus_margin, bbs_relative_to_scaled_image, T2 = scale_image(roll_corrected_image_plus_margin,
                                                           target,
                                                           bbs_relative_to_roll_corrected_image)

    # plt.imsave('/home/gilsh/tmp/for_elad/dump/after_scale.png', scaled_roll_corrected_image_plus_margin/255)
    T10 = np.matmul(T1, T0)
    T210 = np.matmul(T2, T10)

    return scaled_roll_corrected_image_plus_margin, T210, bbs_relative_to_scaled_image[0], bbs_relative_to_scaled_image[1]


########################################################################################################################



def preprocess_image(image, bbs,  roll = 0, roll_support = False, target=BB_SIZE, grayscale=True):

    if not roll_support:
        roll = 0.0

    return preprocess_image_with_roll(image, bbs, roll, grayscale=grayscale, target=target)



########################################################################################################################

def compute_norm_bbs(bbs, T, roll, rgb_padded_rotated, rgb_padded, bbs_plus_margin_input_image):

    # Compute image center
    center_x = rgb_padded_rotated.shape[1] / 2.0
    center_y = rgb_padded_rotated.shape[0] / 2.0

    roll_rad = roll * math.pi / 180.0

    upper_left = [bbs[0], bbs[1], 1]
    down_right = [bbs[2], bbs[3], 1]

    s_x = (rgb_padded.shape[1]) / (bbs_plus_margin_input_image[2] - bbs_plus_margin_input_image[0] + 1)
    s_y = (rgb_padded.shape[0]) / (bbs_plus_margin_input_image[3] - bbs_plus_margin_input_image[1] + 1)


    w = (bbs[2] - bbs[0] + 1) * s_x
    h = (bbs[3] - bbs[1] + 1) * s_y

    # Aspect ratio transformation
    # w_tag / h_tag = pow(w / h, cos(2*roll))

    ar_out = (w / h) ** math.cos(2*roll_rad)

    # Area conversion:
    # w_tag * h_tag = w * h

    w_tag = math.sqrt(w * h * ar_out)
    h_tag = w * h / w_tag

    min_x = center_x - w_tag / 2
    min_y = center_y - h_tag / 2

    return (min_x, min_y)



########################################################################################################################


def transform_image_space_to_norm_space(norm_image, image_bb, roll):
    """ Take a point in the image and transform it to the normalized coordinate system """
    # First translate the point to the FD bounding box frame

    tx = -image_bb[0]
    ty = -image_bb[1]

    Tr = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])


    norm_image_height = norm_image.shape[0]
    norm_image_width = norm_image.shape[1]

    cx = float(norm_image_width) / float((image_bb[2] - image_bb[0] + 1))
    cy = float(norm_image_height) / float((image_bb[3] - image_bb[1] + 1))

    Tc = np.array([[cx,0,0],
                  [0, cy, 0],
                  [0, 0, 1]])

    center_x = (norm_image.shape[1]) / 2.0
    center_y = (norm_image.shape[0]) / 2.0

    M = cv2.getRotationMatrix2D((center_x, center_y), -roll, 1.0)

    # We rotate the whole normalized image (with margins)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((norm_image_height * sin) + (norm_image_width * cos))
    nH = int((norm_image_height * cos) + (norm_image_width * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - center_x
    M[1, 2] += (nH / 2) - center_y

    rotated_image = cv2.warpAffine(norm_image, M, (nW, nH))
    # Concatenate all transformations together:
    M_homo = np.vstack([M, [0, 0, 1]])
    Tcx = np.matmul(Tc, Tr)
    Tcxm = np.matmul(M_homo, Tcx)

    return rotated_image, Tcxm

########################################################################################################################


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))



def transform_points(pts, T):
    """ Apply the inverse transformation on the points collection """

    homo_points = np.array([(x, y, 1) for (y, x) in pts])

    t_points = np.array([T.dot(v) for v in homo_points])
    swap = np.array([(x,y) for (y,x,z) in t_points])
    return swap

########################################################################################################################

def preprocess_image_menpo(image, reference_shape, roll = 0, roll_support = False, target=BB_SIZE, margin=0.3,
                           prev_frame_init=False, grayscale = False):
    im = image.pixels.transpose(1, 2, 0).astype(np.float32) * 255

    im = im.astype(np.uint8)
    lms_bb = image.landmarks['bb'].lms.points
    bbs = [lms_bb[0, 1], lms_bb[0, 0], lms_bb[2, 1], lms_bb[2, 0]]

    im_new, Tmat, min_x, min_y = preprocess_image(im, bbs, roll, roll_support, grayscale=grayscale)
    #print('in preprocess_image_menpo,1: im.shape = '+str(im_new.shape))

    image_new = menpo.image.Image.init_from_channels_at_back(im_new.astype(image.pixels[0,0,0]) / 255.0)

    # we use Tmat instead of bb so this code is (hopefully) obsolete
    image_new.landmarks['__initial'] = reference_shape
    ref_pts = image_new.landmarks['__initial'].lms.points
    image_new.landmarks['__initial'].lms.points = np.array([(y * BB_SIZE/INITIAL_POINTS_BB_SIZE + min_y, x * BB_SIZE/INITIAL_POINTS_BB_SIZE + min_x) for (y, x) in ref_pts])

    image_new.path = image.path

    return image_new, Tmat, min_x, min_y


def resize_x2_x2(im):
    new_im = np.zeros([im.shape[0] / 2, im.shape[1] / 2])
    im_int = np.array(im, dtype=np.int32)
    for y in range(new_im.shape[0]):
        for x in range(new_im.shape[1]):
            avg_upper = np.floor((im_int[y * 2, x * 2] + im_int[y * 2, x * 2 + 1] + 1) / 2)
            avg_lower = np.floor((im_int[y * 2 + 1, x * 2] + im_int[y * 2 + 1, x * 2 + 1] + 1) / 2)
            new_im[y, x] = np.floor((avg_upper + avg_lower + 1) / 2)
    return new_im


def resize_x2_x1(im):
    new_im = np.zeros([im.shape[0] / 2, im.shape[1]])
    im_int = np.array(im, dtype=np.int32)
    for y in range(new_im.shape[0]):
        for x in range(new_im.shape[1]):
            new_im[y, x] = np.floor((im_int[y * 2, x] + im_int[y * 2 + 1, x] + 1) / 2)
    return new_im


def resize_bilinear(input, bbs, is_yuv422):
    padded_input = np.zeros([input.shape[0] + 1, input.shape[1] + 1])
    padded_input[0:-1, 0:-1] = input
    # copy vertical line
    padded_input[:-1, padded_input.shape[1] - 1] = input[:, input.shape[1] - 1]
    # copy vertical line
    padded_input[padded_input.shape[0] - 1, :-1] = input[input.shape[0] - 1, :]

    input_h, input_w = input.shape
    if is_yuv422:
        input_w = input_w * 2
    output_w = bbs[2] - bbs[0] + 1
    output_h = bbs[3] - bbs[1] + 1
    ratio_w = float(input_w) / output_w
    ratio_h = float(input_h) / output_h
    output = np.zeros([output_h, output_w])
    for x in range(output_w):
        x_pos = (x + 0.5) * ratio_w - 0.5
        x_in1 = int(x_pos)
        x_in2 = int(np.ceil(x_pos))
        x_m1 = x_in2 - x_pos
        x_m2 = 1.0 - x_m1
        if is_yuv422:
            x_in1 = int(np.floor(x_in1 / 2))
            x_in2 = int(np.floor(x_in2 / 2))
        for y in range(output_h):
            y_pos = (y + 0.5) * ratio_h - 0.5
            y_in1 = int(y_pos)
            y_in2 = int(np.ceil(y_pos))
            y_m1 = y_in2 - y_pos
            y_m2 = 1.0 - y_m1
            # output[y, x] = (input[y_in1, x_in1] * y_m1 + input[y_in2, x_in1] * y_m2) * x_m1 + (input[y_in1, x_in2] * y_m1 + input[y_in2, x_in2] * y_m2) * x_m2
            output[y, x] = (padded_input[y_in1, x_in1] * y_m1 + padded_input[y_in2, x_in1] * y_m2) * x_m1 + (
                                                                                                            padded_input[
                                                                                                                y_in1, x_in2] * y_m1 +
                                                                                                            padded_input[
                                                                                                                y_in2, x_in2] * y_m2) * x_m2
    # if type(input[0,0]) == np.uint8:
    #     output = np.uint8(output)
    return output


def apply_bbs_transform(pts, bbs, bbs_n, is_pts=True):
    factor_x = float(bbs_n[2] - bbs_n[0] + 1) / (bbs[2] - bbs[0] + 1)
    factor_y = float(bbs_n[3] - bbs_n[1] + 1) / (bbs[3] - bbs[1] + 1)
    if is_pts:
        pts[:, 1] = (pts[:, 1] - bbs[0]) * factor_x + bbs_n[0]
        pts[:, 0] = (pts[:, 0] - bbs[1]) * factor_y + bbs_n[1]
    else:
        pts[0::2] = (pts[0::2] - bbs[0]) * factor_x + bbs_n[0]
        pts[1::2] = (pts[1::2] - bbs[1]) * factor_y + bbs_n[1]

    return pts


########################################################################################################################


def rescale_bb(bb, factor_w, factor_h):
    bbs_m_bbs_origin_w = bb[2] - bb[0] + 1
    bbs_m_bbs_origin_h = bb[3] - bb[1] + 1
    bbs_m_bbs_origin_norm_start_x = bb[0] * factor_w
    bbs_m_bbs_origin_norm_start_y = bb[1] * factor_h
    bbs_m_bbs_origin_norm_end_x = bbs_m_bbs_origin_norm_start_x + bbs_m_bbs_origin_w * factor_w - 1
    bbs_m_bbs_origin_norm_end_y = bbs_m_bbs_origin_norm_start_y + bbs_m_bbs_origin_h * factor_h - 1
    out_bb = [bbs_m_bbs_origin_norm_start_x, bbs_m_bbs_origin_norm_start_y, bbs_m_bbs_origin_norm_end_x,
              bbs_m_bbs_origin_norm_end_y]
    return out_bb


########################################################################################################################



