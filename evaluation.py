
import ntpath
import os
from os.path import relpath
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import matplotlib.patches as patches
from create_face_rect import read_rect_from_file
from create_face_rect import compute_intersection_over_union
import cv2
import json
import glob
from contour import contour_distance
from contour import contour_length
import math
#import cProfile

g_output_dir = "/tmp/"
g_dump_files = True
g_metric = "points_distance"
g_normalization = 'ocular'

########################################################################################################################

def get_indices_per_flm_group():
    indices = {
        'jaw': (0, 18),
        'browL': (19, 28),
        'browR': (29, 38),
        'noseBridge': (39, 41),
        'eyeL': (55, 66),
        'eyeR': (67, 78),
        'nose': (43, 53),
        'mouth': (79, 98),
        'wellDef': (55, 73, 59, 85),
        'nonJaw': [(19, 28), (33, 38), (67, 78), (39, 55), (59, 95)],
        'intOcc': (55, 73),
        'pupils': (99, 100),
        'other':(42, 54)
    }

def get_indices_per_flm_group(format):
    """ get indices for flm group (chin, eyes etc.) per flm format (68, 99 etc.)"""
    if format == 68:
        indices = {
            'patchless': (0, 0),
            'patched': (0,0),
            'pupils': (0,0),
            'middle_nose': (0, 0),
            'periferial_nose': (0,0),
            'inner_mouth': (0,0),
            'outer_mouth': (0, 0),
            'all': (0,67),
            'jaw': (0,16),
            'browL': (17,21),
            'browR': (22,26),
            'eyeL': (36, 41),
            'eyeR': (42, 47),
            'nose': (27, 35),
            'mouth': (48, 67),
            'wellDef': (36, 39, 42, 45),
            'nonJaw': [(17,21), (22,26), (42, 47), (27, 35),(48, 67)],
            'intOcc': (36, 45)
        }
        return indices
    if format == 98:
        indices = {
            'patchless': (0, 0),
            'patched': (0,0),
            'pupils': (0,0),
            'middle_nose': (0, 0),
            'periferial_nose': (0,0),
            'inner_mouth': (0,0),
            'outer_mouth': (0, 0),
            'all': (0,97),
            'jaw': (0,16),
            'browL': (17,21),
            'browR': (22,26),
            'eyeL': (36, 41),
            'eyeR': (42, 47),
            'nose': (27, 35),
            'mouth': (48, 67),
            'wellDef': (36, 39, 42, 45),
            'nonJaw': [(17,21), (22,26), (42, 47), (27, 35),(48, 67)],
            'intOcc': (60, 72)
        }
        return indices


########################################################################################################################

def read_rect(rect_file):
   left, top, right, bottom = read_rect_from_file(rect_file)
   #file = open(rect_file, 'r')
   return [left, top, right - left, bottom - top]


########################################################################################################################

def read_pts(filename):
    #open the pts file
    points = []
    pts_file = open(filename,'r')
    pts_file.readline()
    l = pts_file.readline().split(' ')
    num_points = int(l[-1])
    pts_file.readline()
    for i in range(num_points):
        line = pts_file.readline().split(" ")
        points.append( (float(line[0]) , float(line[1]) ))


    pts_file.close()
    return points


########################################################################################################################


def compute_pts_bb(gt_pts):
    xs = np.array([x for (x, y) in gt_pts])
    ys = np.array([y for (x, y) in gt_pts])
    right = max(xs)
    left = min(xs)
    top = min(ys)
    bottom = max(ys)

    return left, top, right, bottom



########################################################################################################################

def compute_rect_error(gt_pts, rect_file_name):
    gt_left, gt_top, gt_right, gt_bottom = compute_pts_bb(gt_pts)

    estimated_rect = read_rect(rect_file_name)
    est_top = estimated_rect[1]
    est_left = estimated_rect[0]
    est_right = est_left + estimated_rect[2]
    est_bottom = est_top + estimated_rect[3]

    est_rect = (est_left, est_top, est_right, est_bottom)
    gt_rect = (gt_left, gt_top, gt_right, gt_bottom)
    iot = compute_intersection_over_union(est_rect, gt_rect)

    return iot



########################################################################################################################


# we receive a directory list. At each directory we expect to find flm estimation results (one pts file per face)
# The following error data is calculated for each test set:
# 1. normalised error per flm point, mean and std per point
# 2. avg and std on all faces for 68 points
# 3. avg and std on all faces for 51 points
# 4. maximum error per face (68 and 51 points)
# 5. Cumulative Error Distribution graph for 68 and 51 points - merge graphs for all directories

# mean_jaw = np.mean(face_error_list[flm_indices['jaw'][0]:flm_indices['jaw'][1] + 1])
def get_relevant_inds(part_indices):
    return np.concatenate([range(i, j) for i, j in part_indices])

def get_relevant_flms(face_error_list, part_indices):
    # return np.take(face_error_list, get_relevant_inds(part_indices))
    return np.take(face_error_list, part_indices)


########################################################################################################################

def compute_error_error(estimated_pts_file, gt_pts, estimated_pts, interoccular_distance):
    face_error_list = []
    norm_error_vec = (np.array(gt_pts) - np.array(estimated_pts)) / interoccular_distance

    for point_gt, point_est in zip(gt_pts, estimated_pts):
        norm_dst = distance.euclidean(point_gt, point_est) / interoccular_distance
        face_error_list.append(norm_dst)

    flm_count = len(face_error_list)
    mean_all_flm = np.mean(face_error_list)

    estimated_error_error = -1
    estimated_error = -1
    estimated_error_file_name = estimated_pts_file[:-3] + 'txt'
    if os.path.isfile(estimated_error_file_name):
        with open(estimated_error_file_name,'r') as f:
            estimated_error = float(f.readline())
            estimated_error_error = np.abs(estimated_error - mean_all_flm)

    return estimated_error_error, estimated_error

########################################################################################################################

def calculate_contour_error_per_face(image_file, gt_pts_file, estimated_pts_file, rect_file_name, dataset_name, flm_count, step_loss = False, threshold = 0):
    gt_pts = np.array(read_pts(gt_pts_file), dtype=np.int32)
    # gil
    estimated_pts = np.array(read_pts(estimated_pts_file), dtype=np.int32)
    if flm_count == 10:
        FLM_INDEXES = [55, 58, 61, 64, 100, 67, 70, 73, 76, 99]
        gt_pts = gt_pts[FLM_INDEXES, :]
        if estimated_pts.shape[0] == 165:
            estimated_pts= estimated_pts[FLM_INDEXES,:]


    num_flms = len(gt_pts)
    # gt_contours = get_contours(gt_pts)
    # estimated_contour = get_contours(estimated_pts)
    flm_indices = get_indices_per_flm_group(num_flms)

    interoccular_distance = distance.euclidean(gt_pts[flm_indices['intOcc'][0]], gt_pts[flm_indices['intOcc'][1]])

    estimated_error_error, estimated_error = compute_error_error(estimated_pts_file, gt_pts, estimated_pts, interoccular_distance)

    max_contour_dist = 4.0 * interoccular_distance

    #contour_distance return the average contour distance per sampled point

    mean_jaw_error = contour_distance(gt_pts[flm_indices['jaw']], estimated_pts[flm_indices['jaw']], max_contour_dist) / interoccular_distance
    jaw_length = contour_length(estimated_pts[flm_indices['jaw']]) / interoccular_distance

    mean_left_eye_error = contour_distance(gt_pts[flm_indices['eyeL']], estimated_pts[flm_indices['eyeL']], max_contour_dist) / interoccular_distance
    left_eye_length = contour_length(estimated_pts[flm_indices['eyeL']]) / interoccular_distance

    mean_right_eye_error = contour_distance(gt_pts[flm_indices['eyeR']], estimated_pts[flm_indices['eyeR']], max_contour_dist) / interoccular_distance
    right_eye_length = contour_length(estimated_pts[flm_indices['eyeR']]) / interoccular_distance

    mean_left_eyebrow_error = contour_distance(gt_pts[flm_indices['browL']], estimated_pts[flm_indices['browL']], max_contour_dist) / interoccular_distance
    left_eyebrow_length = contour_length(estimated_pts[flm_indices['browL']]) / interoccular_distance

    mean_right_eyebrow_error = contour_distance(gt_pts[flm_indices['browR']], estimated_pts[flm_indices['browR']], max_contour_dist) / interoccular_distance
    right_eyebrow_length = contour_length(estimated_pts[flm_indices['browR']]) / interoccular_distance

    mean_inner_mouth_error = contour_distance(gt_pts[flm_indices['inner_mouth']], estimated_pts[flm_indices['inner_mouth']], max_contour_dist) / interoccular_distance
    inner_mouth_length = contour_length(estimated_pts[flm_indices['inner_mouth']]) / interoccular_distance

    mean_outer_mouth_error = contour_distance(gt_pts[flm_indices['outer_mouth']], estimated_pts[flm_indices['outer_mouth']], max_contour_dist) / interoccular_distance
    outer_mouth_length = contour_length(estimated_pts[flm_indices['outer_mouth']]) / interoccular_distance

    mean_periferial_nose_error = contour_distance(gt_pts[flm_indices['periferial_nose']], estimated_pts[flm_indices['periferial_nose']], max_contour_dist) / interoccular_distance
    periferial_nose_length = contour_length(estimated_pts[flm_indices['periferial_nose']]) / interoccular_distance

    mean_middle_nose_error = contour_distance(gt_pts[flm_indices['middle_nose']], estimated_pts[flm_indices['middle_nose']], max_contour_dist) / interoccular_distance
    middel_nose_length = contour_length(estimated_pts[flm_indices['middle_nose']]) / interoccular_distance

    AVERAGE_INTEROCCULAR_DISTANCE = 63
    AVERAGE_IRIS_DIAMETER = 10.2
    AVERAGE_IRIS_CIRCUMFERENCE = math.pi * AVERAGE_IRIS_DIAMETER
    norm_iris_circumference = AVERAGE_IRIS_CIRCUMFERENCE / AVERAGE_INTEROCCULAR_DISTANCE

    pupil_left_error = distance.euclidean(gt_pts[flm_indices['left_pupil']], estimated_pts[flm_indices['left_pupil']]) / interoccular_distance
    pupil_right_error = distance.euclidean(gt_pts[flm_indices['right_pupil']], estimated_pts[flm_indices['right_pupil']]) / interoccular_distance

    # gil
    total_contour_length = jaw_length + left_eye_length + right_eye_length + left_eyebrow_length + \
                           right_eyebrow_length + inner_mouth_length + outer_mouth_length + middel_nose_length + \
                           periferial_nose_length + 2 * norm_iris_circumference

    # total_contour_length = jaw_length + left_eye_length + right_eye_length + left_eyebrow_length + \
    #                        right_eyebrow_length + inner_mouth_length + outer_mouth_length + \
    #                        2 * norm_iris_circumference

    # gil
    mean_all_contour = mean_jaw_error * jaw_length + \
                       mean_left_eye_error * left_eye_length + \
                       mean_right_eye_error * right_eyebrow_length + \
                       mean_left_eyebrow_error * left_eyebrow_length + \
                       mean_right_eyebrow_error * right_eyebrow_length + \
                       mean_inner_mouth_error * inner_mouth_length + \
                       mean_outer_mouth_error * outer_mouth_length + \
                       mean_middle_nose_error * middel_nose_length + \
                       mean_periferial_nose_error * periferial_nose_length + \
                       norm_iris_circumference * pupil_left_error + \
                       norm_iris_circumference * pupil_right_error

    # mean_all_contour = mean_jaw_error * jaw_length + \
    #                    mean_left_eye_error * left_eye_length + \
    #                    mean_right_eye_error * right_eyebrow_length + \
    #                    mean_left_eyebrow_error * left_eyebrow_length + \
    #                    mean_right_eyebrow_error * right_eyebrow_length + \
    #                    mean_inner_mouth_error * inner_mouth_length + \
    #                    mean_outer_mouth_error * outer_mouth_length + \
    #                    norm_iris_circumference * pupil_left_error + \
    #                    norm_iris_circumference * pupil_right_error


    mean_all_contour /= total_contour_length

    mean_non_jaw =     mean_left_eye_error * left_eye_length + \
                       mean_right_eye_error * right_eyebrow_length + \
                       mean_left_eyebrow_error * left_eyebrow_length + \
                       mean_right_eyebrow_error * right_eyebrow_length + \
                       mean_inner_mouth_error * inner_mouth_length + \
                       mean_outer_mouth_error * outer_mouth_length + \
                       mean_middle_nose_error * middel_nose_length + \
                       mean_periferial_nose_error * periferial_nose_length + \
                       norm_iris_circumference * pupil_left_error + \
                       norm_iris_circumference * pupil_right_error

    # gilflm_indices['jaw']
    non_jaw_length = left_eye_length + right_eye_length + left_eyebrow_length + \
                           right_eyebrow_length + inner_mouth_length + outer_mouth_length + middel_nose_length + \
                           periferial_nose_length + 2 * norm_iris_circumference


    # non_jaw_length = left_eye_length + right_eye_length + left_eyebrow_length + \
    #                        right_eyebrow_length + inner_mouth_length + outer_mouth_length + \
    #                        2 * norm_iris_circumference

    mean_non_jaw /= non_jaw_length
    rect_error = compute_rect_error(gt_pts, rect_file_name)
    error_data_for_face = {
        'estimated_error': estimated_error,
        'estimated_error_error': estimated_error_error,
        'rect_error': rect_error,
        # 'norm_error_vec': norm_error_vec,
        'flm_count': num_flms,
        'mean_non_jaw': mean_non_jaw,
        # 'max_all_flm': max_all,
        'mean_all_flm': mean_all_contour,
        'mean_jaw': mean_jaw_error,
        'mean_middle_nose': mean_middle_nose_error,
        'mean_periferal_nose': mean_periferial_nose_error,
        'mean_left_eye': mean_left_eye_error,
        'mean_right_eye': mean_right_eye_error,
        'mean_left_brow': mean_left_eyebrow_error,
        'mean_right_brow': mean_right_eyebrow_error,
        'mean_inner_mouth': mean_inner_mouth_error,
        'mean_outer_mouth': mean_outer_mouth_error,
        # 'rect_error': rect_error,
        'gt_data_file': gt_pts_file,
        'estimated_data_file': estimated_pts_file,
        'rect_data_file': rect_file_name,
        'mean_pupils': (pupil_left_error + pupil_right_error) / 2.0
        # 'mean_patched': mean_patched,
        # 'mean_patchless': mean_patchless
    }

    if g_dump_files:
        # image_file = gt_pts_file[0:-3] + 'jpg'
        dump_debug_image_to_file(image_file, gt_pts, estimated_pts, rect_file_name, dataset_name, mean_all_contour)

    return error_data_for_face


########################################################################################################################

def compute_interpupil_dist(gt_pts, flm_indices):
    left_eye_indexes = range(flm_indices['eyeL'][0], flm_indices['eyeL'][-1] + 1)
    right_eye_indexes = range(flm_indices['eyeR'][0], flm_indices['eyeR'][-1] + 1)

    left_eye_points = np.array([gt_pts[i] for i in left_eye_indexes])
    right_eye_points = np.array([gt_pts[i] for i in right_eye_indexes])
    left_eye_center = np.average(left_eye_points, axis=0)
    right_eye_center = np.average(right_eye_points, axis=0)
    dist = np.linalg.norm(left_eye_center - right_eye_center)
    return dist

########################################################################################################################


def calculate_error_per_face(image_file, gt_pts_file, estimated_pts_file, rect_file_name, dataset_name, step_loss = False, threshold = 0):

    gt_pts = read_pts(gt_pts_file)
    num_flms = len(gt_pts)
    estimated_pts = read_pts(estimated_pts_file)


    flm_indices = get_indices_per_flm_group(num_flms)
    if g_normalization == 'ocular':
        norm_distance = distance.euclidean(gt_pts[flm_indices['intOcc'][0]], gt_pts[flm_indices['intOcc'][1]])
    elif g_normalization == 'pupil':
        norm_distance = compute_interpupil_dist(gt_pts, flm_indices)
    else:
        raise Exception('Unsupported normalization')


    face_error_list = []
    norm_error_vec = (np.array(gt_pts) - np.array(estimated_pts)) / norm_distance

    estimated_error_error, estimated_error = compute_error_error(estimated_pts_file, gt_pts, estimated_pts,
                                                                 norm_distance)
    for point_gt, point_est in zip(gt_pts, estimated_pts):
        norm_dst = distance.euclidean(point_gt, point_est) / norm_distance
        norm_dst = norm_dst if norm_dst > threshold else 0
        face_error_list.append(norm_dst)

    flm_count = len(face_error_list)
    max_all = np.max(face_error_list)
    mean_all_flm = np.mean(face_error_list)

    estimated_error_error = -1
    estimated_error_file_name = estimated_pts_file[:-3] + 'txt'
    if os.path.isfile(estimated_error_file_name):
        with open(estimated_error_file_name,'r') as f:
            estimated_error_error = np.abs(float(f.readline()) - mean_all_flm)


    # mean_jaw = np.mean(face_error_list[flm_indices['jaw'][0]:flm_indices['jaw'][1] + 1])
    # mean_nose = np.mean(face_error_list[flm_indices['nose'][0]:flm_indices['nose'][1] + 1])
    # mean_left_eye = np.mean(face_error_list[flm_indices['eyeL'][0]:flm_indices['eyeL'][1] + 1])
    # mean_right_eye = np.mean(face_error_list[flm_indices['eyeR'][0]:flm_indices['eyeR'][1] + 1])
    # mean_left_brow = np.mean(face_error_list[flm_indices['browL'][0]:flm_indices['browL'][1] + 1])
    # mean_right_brow = np.mean(face_error_list[flm_indices['browR'][0]:flm_indices['browR'][1] + 1])
    # mean_mouth = np.mean(face_error_list[flm_indices['mouth'][0]:flm_indices['mouth'][1] + 1])
    # mean_pupils = np.mean(face_error_list[flm_indices['pupils'][0]:flm_indices['pupils'][1] + 1])

    mean_jaw = np.mean(get_relevant_flms(face_error_list, flm_indices['jaw']))
    mean_nose = np.mean(get_relevant_flms(face_error_list, flm_indices['nose']))
    mean_left_eye = np.mean(get_relevant_flms(face_error_list, flm_indices['eyeL']))
    mean_right_eye = np.mean(get_relevant_flms(face_error_list, flm_indices['eyeR']))
    mean_left_brow = np.mean(get_relevant_flms(face_error_list, flm_indices['browL']))
    mean_right_brow = np.mean(get_relevant_flms(face_error_list, flm_indices['browR']))
    mean_mouth = np.mean(get_relevant_flms(face_error_list, flm_indices['mouth']))
    mean_pupils = np.mean(get_relevant_flms(face_error_list, flm_indices['pupils']))

    mean_middle_nose_error = np.mean(get_relevant_flms(face_error_list, flm_indices['middle_nose']))
    mean_periferial_nose_error = np.mean(get_relevant_flms(face_error_list, flm_indices['periferial_nose']))
    mean_inner_mouth_error = np.mean(get_relevant_flms(face_error_list, flm_indices['inner_mouth']))
    mean_outer_mouth_error = np.mean(get_relevant_flms(face_error_list, flm_indices['outer_mouth']))

    # with/without patch - pointwise (not sequential)
    mean_patched = np.mean([face_error_list[i] for i in flm_indices['patched']])
    mean_patchless = np.mean([face_error_list[i] for i in flm_indices['patchless']])

    all_indexes = flm_indices['all']
    jaw_indexes = flm_indices['jaw']

    non_jaw_indexes = list(set(all_indexes) - set(jaw_indexes))
    mean_non_jaw = np.mean([face_error_list[i] for i in non_jaw_indexes])
    rect_error = compute_rect_error(gt_pts, rect_file_name)
    error_data_for_face = {
        'estimated_error': estimated_error,
        'estimated_error_error': estimated_error_error,
        'norm_error_vec': norm_error_vec,
        'flm_count': flm_count,
        'max_all_flm': max_all,
        'mean_all_flm': mean_all_flm,
        'mean_jaw': mean_jaw,
        'mean_nose': mean_nose,
        'mean_left_eye': mean_left_eye,
        'mean_right_eye': mean_right_eye,
        'mean_left_brow': mean_left_brow,
        'mean_right_brow': mean_right_brow,
        'mean_mouth': mean_mouth,
        'mean_non_jaw': mean_non_jaw,
        'rect_error': rect_error,
        'gt_data_file': gt_pts_file,
        'estimated_data_file': estimated_pts_file,
        'rect_data_file': rect_file_name,
        'mean_pupils': mean_pupils,
        'mean_patched': mean_patched,
        'mean_patchless': mean_patchless,

        'mean_middle_nose': mean_middle_nose_error,
        'mean_periferal_nose': mean_periferial_nose_error,
        'mean_inner_mouth': mean_inner_mouth_error,
        'mean_outer_mouth': mean_outer_mouth_error,

    }

    # dump debug image
    g_dump_files = False
    if g_dump_files:
        # image_file = gt_pts_file[0:-3] + 'jpg'
        dump_debug_image_to_file(image_file, gt_pts, estimated_pts, rect_file_name, dataset_name, mean_all_flm)
#        dump_stats(image_file, dataset_name, error_data_for_face)


    return error_data_for_face, face_error_list



########################################################################################################################

def dump_stats(image_file, dataset_name, error_data_for_face):

    base_name = ntpath.basename(image_file)
    output_dir = g_output_dir + '/' + dataset_name

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    path = g_output_dir + '/' + dataset_name + '/' + str(error) + '_' +  base_name[0:-3] + 'json'

    with open(path, 'w') as outfile:
        json.dump(error_data_for_face, outfile)


########################################################################################################################

def draw_points_on_image(image, points, point_size, color):
    for x,y in points:
        cv2.circle(image, (int(x) , int(y)), int(point_size), color, -1)




########################################################################################################################

def dump_debug_image_to_file(image_file, gt_pts, estimated_pts, rect_file_name, dataset_name, error):

    image = cv2.imread(image_file)

    estimated_rect = read_rect(rect_file_name)
    width = estimated_rect[2]
    est_top = estimated_rect[1]
    est_left = estimated_rect[0]
    est_right = est_left + estimated_rect[2]
    est_bottom = est_top + estimated_rect[3]

    point_size = 0.01 * (width)
    draw_points_on_image(image, estimated_pts, point_size, color=(0,0,255))
    draw_points_on_image(image, gt_pts, point_size, color=(0,255,0))

    cv2.rectangle(image, (int(est_left), int(est_top)), (int(est_right), int(est_bottom)), (0, 255, 0), 3)


    base_name = ntpath.basename(image_file)
    output_dir = g_output_dir + '/' + dataset_name

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    path = g_output_dir + '/' + dataset_name + '/' + str(error) + '_' +  base_name[0:-3] + 'jpg'

    print ('write image: ' + path)
    cv2.imwrite(path, image)

########################################################################################################################
def calc_ced(extracted_data, error_table, at_recall = -1):
        total_faces = len(error_table)
        index = 0

        estimated_error_error = np.sort([x['estimated_error_error'] for x in error_table])
        extracted_data['ced_estimated_error'] = []
        for error in estimated_error_error:
            fraction_of_faces_with_lower_error = float(index) / float(len(estimated_error_error))
            extracted_data['ced_estimated_error'].append((error, fraction_of_faces_with_lower_error))
            index+=1

        extracted_data['ced_recall_vs_precision'] = []
        estimated_error = np.sort([x['estimated_error'] for x in error_table])
        error_subset = []
        if at_recall != -1:
            for index, error in enumerate(estimated_error):
                if error == -1:
                    continue
                errors_smaller_than_error = [x['mean_all_flm'] for x in error_table if x['estimated_error'] <= error]
                errors_smaller_than_error = np.sort(errors_smaller_than_error)
                recall = float(len(errors_smaller_than_error)) / float(len(estimated_error))
                if np.abs(at_recall - recall) < 0.001:
                    error_subset = errors_smaller_than_error

                # j = (errors_smaller_than_error < 0.08).sum()
                # ced_008 = float(j) / float(len(errors_smaller_than_error))
                # avg = (errors_smaller_than_error.sum()) / float(len(errors_smaller_than_error))
                # extracted_data['ced_recall_vs_precision'].append((recall, ced_008))

        if len(error_subset) > 0:
            all_data = error_subset
        else:
            all_data = np.sort([x['mean_all_flm'] for x in error_table])

        non_jaw_data = np.sort([x['mean_non_jaw'] for x in error_table])
        rect_error = np.sort([x['rect_error'] for x in error_table])
        extracted_data['ced_data_all'] = []
        index = 1

        for error in all_data:
            fraction_of_faces_with_lower_error = float(index) / float(len(all_data))
            extracted_data['ced_data_all'].append((error, fraction_of_faces_with_lower_error))
            index+=1


        index = 1
        extracted_data['ced_non_jaw'] = []
        for error in non_jaw_data:
            fraction_of_faces_with_lower_error = float(index) / float(len(non_jaw_data))
            extracted_data['ced_non_jaw'].append((error, fraction_of_faces_with_lower_error))
            index += 1

        index = 1

        extracted_data['ced_rect'] = []
        for error in rect_error:
            fraction_of_faces_with_lower_error = float(index) / float(len(non_jaw_data))
            extracted_data['ced_rect'].append((error, fraction_of_faces_with_lower_error))
            index += 1






########################################################################################################################

def calc_data_for_visualisation(extracted_data, error_table):
        #sort the error table according to maximum error (column 68)
        # error_table.sort(key = lambda row: row[68])
        error_table = sorted(error_table, key = lambda row: row['mean_all_flm'])

        num_of_faces = len(error_table)
        sample_count = 4
        extracted_data['best_results'] = []
        for i in range(sample_count):
            extracted_data['best_results'].append( error_table[i] )

        extracted_data['worst_results'] = []
        for i in range((num_of_faces - sample_count),num_of_faces):
            extracted_data['worst_results'].append( error_table[i] )

        number_of_random_samples = 10
        random_selection_form_error_table = random.sample(error_table, number_of_random_samples)
        extracted_data['random_results'] = []
        for face_row in random_selection_form_error_table:
            extracted_data['random_results'].append( face_row )

########################################################################################################################

def calc_bar(extracted_data, error_table, threshold):
    bar_data = {}
    bar_data['LEye'] = len([x for x in error_table if x['mean_left_eye'] < threshold]) / float(len(error_table))
    bar_data['REye'] = len([x for x in error_table if x['mean_right_eye'] < threshold]) / float(len(error_table))
    bar_data['LBrow'] = len([x for x in error_table if x['mean_left_brow'] < threshold]) / float(len(error_table))
    bar_data['RBrow'] = len([x for x in error_table if x['mean_right_brow'] < threshold]) / float(len(error_table))
    # bar_data['Nose'] = len([x for x in error_table if x['mean_nose'] < threshold]) / float(len(error_table))
    # bar_data['Mouth'] = len([x for x in error_table if x['mean_mouth'] < threshold]) / float(len(error_table))
    bar_data['Jaw'] = len([x for x in error_table if x['mean_jaw'] < threshold]) / float(len(error_table))
    bar_data['All'] = len([x for x in error_table if x['mean_all_flm'] < threshold]) / float(len(error_table))
    bar_data['Non Jaw'] = len([x for x in error_table if x['mean_non_jaw'] < threshold]) / float(len(error_table))
    bar_data['Pupils'] = len([x for x in error_table if x['mean_pupils'] < threshold]) / float(len(error_table))
    # bar_data['Patched'] = len([x for x in error_table if x['mean_patched'] < threshold]) / float(len(error_table))
    # bar_data['Patchless'] = len([x for x in error_table if x['mean_patchless'] < threshold]) / float(len(error_table))
    #

    ret = bar_data
    return ret
    # bar_data_per_thresh = {}
    # bar_data_per_thresh[threshold] = bar_data
    # extracted_data['bar_data'] = bar_data_per_thresh

def read_pts(filename):
    # open the pts file
    points = []
    pts_file = open(filename, 'r')
    pts_file.readline()
    l = pts_file.readline().rstrip().split(' ')
    num_points = int(l[-1])
    pts_file.readline()
    for i in range(num_points):
        line = pts_file.readline().split(" ")
        points.append((float(line[0]), float(line[1])))

    pts_file.close()
    return points

def calc_error_per_flm(error_list, dataset_name, verbose = True):
    num_flm = len(error_list[0])
    mean = np.zeros((num_flm,))
    std = np.zeros((num_flm,))
    for i in range(num_flm):
        v = [it[i] for it in error_list]
        mean[i] = np.mean(v)
        std[i] = np.std(v)

    filename = g_output_dir + '/' + 'error_per_flm_' + dataset_name.replace(' ','_') + '.txt'
    indices = get_indices_per_flm_group(num_flm)
    noPatch = [i for i in range(num_flm)if i not in indices['patched']]

    # hasPatch = ['T' if i in indices['patched'] else 'F' for i in range(num_flm) ]
    # with open(filename,'wt') as f:
    #     f.writelines('Ind\tMean\tStd\thasPatch\n')
    #     f.writelines(['%d\t%.2f\t%.2f\t%s\n'%elem for elem in zip(range(num_flm), mean, std, hasPatch)])

    # draw on image
    imname = 'w:/Geunhee_data/001_normal.jpg'
    pts_name = 'W:/300W_LP/semi_frontal/experiments/nfd_3d_with_extreme/res_geunhee_28800/001_normal.pts'
    a = read_pts(pts_name)
    a = np.array(a)
    im = plt.imread(imname)
    plt.imshow(im)
    max_err = max(mean)
    min_err = min(mean)
    err_range = max_err - min_err
    for i in indices['patched']:
        err = mean[i]
        rval = (err - min_err)/err_range
        plt.plot(a[i, 0], a[i, 1], 'o', color = [rval, 0, 1-rval], ms=10)
    for i in indices['patched']:
        err = mean[i]
        rval = (err - min_err) / err_range
        plt.plot(a[i, 0], a[i, 1], 'x', color=[rval, 0, 1 - rval], ms=10)

    return mean, std


########################################################################################################################

def extract_error_data(error_table, recall_rate):
    extracted_data = {}
    aggregate_error_vec = []
    # aggregate_error_vec = aggregate_error_vec.append(x['norm_error_vec'] for x in error_table)
    jaw_error_vec = []
    non_jaw_error_vec = []
    nose_error_vec = []
    left_eye_error_vec = []
    right_eye_error_vec = []
    left_brow_error_vec = []
    right_brow_error_vec = []
    mouth_error_vec = []
    middel_nose_error_vec = []
    inner_mouth_error_vec = []
    outer_mouth_error_vec = []
    middle_nose_error_vec = []
    periferal_nose_error_vec = []
    pupils_error_vec = []
    estimated_error_error_vec = []

    THRESH = 0.1
    indices = get_indices_per_flm_group(error_table[0]['flm_count'])
    # gil
    # for it in error_table:
    #     for p in get_relevant_flms(it['norm_error_vec'],indices['jaw']):
    #         if np.linalg.norm(p) < THRESH:
    #             jaw_error_vec.append(p)
    #     for p in get_relevant_flms(it['norm_error_vec'], indices['browL']):
    #         if np.linalg.norm(p) < THRESH:
    #             left_brow_error_vec.append(p)
    #     for p in get_relevant_flms(it['norm_error_vec'],indices['browR']):
    #         if np.linalg.norm(p) < THRESH:
    #             right_brow_error_vec.append(p)
    #     for p in get_relevant_flms(it['norm_error_vec'],indices['eyeL']):
    #         if np.linalg.norm(p) < THRESH:
    #             left_eye_error_vec.append(p)
    #     for p in get_relevant_flms(it['norm_error_vec'],indices['eyeR']):
    #         if np.linalg.norm(p) < THRESH:
    #             right_eye_error_vec.append(p)
    #     for p in get_relevant_flms(it['norm_error_vec'],indices['nose']):
    #         if np.linalg.norm(p) < THRESH:
    #             nose_error_vec.append(p)
    #     for p in get_relevant_flms(it['norm_error_vec'],indices['mouth']):
    #         if np.linalg.norm(p) < THRESH:
    #             mouth_error_vec.append(p)

    for it in error_table:
        non_jaw_error_vec.append([it['mean_non_jaw']])
        jaw_error_vec.append(it['mean_jaw'])
        left_brow_error_vec.append(it['mean_left_brow'])
        right_brow_error_vec.append(it['mean_right_brow'])
        right_eye_error_vec.append(it['mean_right_eye'])
        left_eye_error_vec.append(it['mean_left_eye'])
        middle_nose_error_vec.append(it['mean_middle_nose'])
        periferal_nose_error_vec.append(it['mean_periferal_nose'])
        pupils_error_vec.append(it['mean_pupils'])
        inner_mouth_error_vec.append(it['mean_inner_mouth'])
        outer_mouth_error_vec.append(it['mean_outer_mouth'])
        estimated_error_error_vec.append(it['estimated_error_error'])


    extracted_data['estimated_error_error'] = estimated_error_error_vec
    extracted_data['jaw_error_vec'] = jaw_error_vec
    extracted_data['left_brow_error_vec'] = left_brow_error_vec
    extracted_data['right_brow_error_vec'] = right_brow_error_vec
    extracted_data['right_eye_error_vec'] = right_eye_error_vec
    extracted_data['left_eye_error_vec'] = left_eye_error_vec
    extracted_data['inner_mouth_error_vec'] = inner_mouth_error_vec
    extracted_data['outer_mouth_error_vec'] = outer_mouth_error_vec
    extracted_data['middle_nose_error_vec'] = middle_nose_error_vec
    extracted_data['periferal_nose_error_vec'] = periferal_nose_error_vec


    extracted_data['total_avg_all'] = np.mean([x['mean_all_flm'] for x in error_table])
    extracted_data['total_std_all'] = np.std([x['mean_all_flm'] for x in error_table])
    extracted_data['total_avg_non_jaw'] = np.mean([x['mean_non_jaw'] for x in error_table])
    extracted_data['total_std_non_jaw'] = np.std([x['mean_non_jaw'] for x in error_table])
    # 0.448 is the recall rate of Arcsoft on IBUG
    calc_ced(extracted_data, error_table, recall_rate)
#    calc_per_flm(error_table)

    extracted_data['bar_data'] = {}
    extracted_data['bar_data'][0.08] = calc_bar(extracted_data, error_table, 0.08)
    extracted_data['bar_data'][0.05] = calc_bar(extracted_data, error_table, 0.05)
    # calc_data_for_visualisation(extracted_data, error_table)


    return extracted_data



########################################################################################################################



def compute_error_data(estimated_dir_path, input_dir, rect_dir, image_dir, dataset_name, flm_count,
                       step_loss = False, threshold = 0, filenames=None, recall_rate=-1):
    if filenames == None:
        filenames = g_common_filenames

    error_table = []
    error_list = []
    #for filename in os.listdir(estimated_dir_path):
    missing_files = []
    for filename in filenames :
        estimated_pts_file = estimated_dir_path + "/" + filename
        print(filename)

        gt_pts_file = input_dir + "/" + filename
        image_file = image_dir + '/' + filename[0:-3] + 'jpg'
        if rect_dir == "":
            rect_file_name = ""
        else:
            rect_file_name = rect_dir + "/" + filename

        if g_metric != 'contour_distance':
            face_data_row, face_error_list = calculate_error_per_face(image_file, gt_pts_file, estimated_pts_file, rect_file_name, dataset_name, step_loss, threshold)
        else:
            face_data_row = calculate_contour_error_per_face(image_file, gt_pts_file, estimated_pts_file,
                                                                      rect_file_name, dataset_name, flm_count, step_loss, threshold )
        error_table.append(face_data_row)
        # error_list.append(face_error_list)

  #  calc_error_per_flm(error_list, dataset_name)
    extracted_error_data = extract_error_data(error_table, recall_rate)


    error_data_item = {
        'error_table': extracted_error_data,
        'dataset_name': dataset_name
    }
    return error_data_item



########################################################################################################################



def report_error_per_dir(error_data_per_dir):
    for error_data in error_data_per_dir:
        print ("------------------------------------------------------------------------------------------------------\n")
        print ("Error metrics for estimated FLM in " + error_data['dataset_name'] + "\n")
        for i in range(101):
             print ("flm %d (avg = %f, std = %f) " % (i, error_data.avg_std_per_flm[i][0], error_data.avg_std_per_flm[i][1]))

        # print "\n\n\n"
        # print "Average Error all: %f \n" % error_data['total_avg_all']
        # print "STD Error all: %f \n" % error_data['total_std_all']
        # print "Error rate all: %f (the fraction of faces with average FLM error above %f)\n" % (error_data.error_rate_all, error_data.error_thresh_for_error_rate)
        # print "Average Error no jaw: %f \n" % error_data['total_avg_all']
        # print "STD Error no jaw: %f \n" % error_data['total_std_non_jaw']
        # print "Error rate jaw: %f (the fraction of faces with average FLM error above %f)\n" % (error_data.error_rate_non_jaw, error_data.error_thresh_for_error_rate)

        # plot some results

      #  plot_results(error_data, estimated_dir)


########################################################################################################################

def ced_at_err(err, face_ratio, err_pt):
    less_than = [i for i in range(len(err)) if err[i] <= err_pt]
    if not(less_than):
        return 0

    return face_ratio[less_than[-1]]



########################################################################################################################




def print_statistics(error_data_per_dir):
    for error_data in error_data_per_dir:
        short_dir_name = error_data['dataset_name']
        avg_error = error_data['error_table']['total_avg_all']
        std = error_data['error_table']['total_std_all']

        print('Average error for dataset {}: {}, std: {}'.format(short_dir_name, avg_error, std))
########################################################################################################################



def plot_ced(error_data_per_dir, window_title, ced_data_type):

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', '*', 'p', 's', 'x','+', '.']
    plt.figure()
    fig = plt.gcf()
    # fig.canvas.set_window_title('Normalized Point-to-Point Error 68-points')
    fig.canvas.set_window_title(window_title)
    # plt.xlabel('Normalized Point-to-Point Error 68-points')
    plt.xlabel(window_title)
    plt.ylabel('Faces Proportion')
    plt.grid(linestyle='dotted')
    plot_handles = []
    ci = 0
    for error_data in error_data_per_dir:
        x_all_arr = []
        y_all_arr = []

        # short_dir_name = [os.path.basename(d) for d in dir_name ]
        short_dir_name = error_data['dataset_name']
       # np.save(g_output_dir + '/' + short_dir_name + '_ced.npy',
       #         error_data['error_table'][ced_data_type])

        for x,y in error_data['error_table'][ced_data_type]:
            if x < 0.1:
                x_all_arr.append(x)
                y_all_arr.append(y)
        err_006 = ced_at_err(x_all_arr, y_all_arr, 0.06)
        err_008 = ced_at_err(x_all_arr, y_all_arr, 0.08)

        if ced_data_type == 'ced_data_all':
            label = short_dir_name + '\n0.06 val = '+'%.2f'%err_006+', 0.08 val = '+'%.2f'%err_008
        else:
            label = ""

        plt.plot(x_all_arr, y_all_arr, 'bo', marker=markers[ci % len(markers)], linestyle='--',
                 color=colors[ci % len(colors)], label=label, markersize=3,alpha=.6,markeredgecolor=colors[ci % len(colors)])
        # plt.xlim(0.04,0.1)
        #plot_handles.append(pl)
        ci = ci + 1

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
     #         fancybox=True, shadow=True, ncol=5, fontsize='medium')

    leg = plt.legend(loc=0, fontsize='small')
    if ced_data_type == 'ced_data_all':
        leg.get_frame().set_alpha(0.5)

    out_fig_file = g_output_dir + '/' + window_title + '.pdf'
    plt.savefig(out_fig_file)
    # plt.show(block=False)



########################################################################################################################

def plot_bar(error_data_per_dir, title, error_value):
    plt.figure()
    plt.title(title)
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.xlabel('Facial Parts')
    plt.ylabel('Faces Proportion')
    color_table = ['r','b','g','y','c','m','k','w']

    ax = plt.subplot(111)
    column_count = len(error_data_per_dir)
    width = 1.0 / float(column_count)
    width = 0.8 * width
    total_width = width * (column_count - 1)
    start_position = -total_width / 2.0
    i = 0
    for error_data in error_data_per_dir:
        keys = []
        values = []
        short_dir_name = error_data['dataset_name']
        bar_data = error_data['error_table']['bar_data'][error_value]
        for key, value in list(bar_data.items()):
            keys.append(key)
            values.append(value)
        ind = np.arange(len(keys))
        # ax.bar(ind + start_position + width*i , values, width=0.2, align='center', label=short_dir_name)
        ax.bar(ind + start_position + width * i, values, width, align='center', label=short_dir_name, color=color_table[i%column_count])
        # ax.bar(ind , values, width, align='center', label=short_dir_name)
        i += 1
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, fontsize='x-small')
    leg = plt.legend(loc=0, fontsize='small')
    leg.get_frame().set_alpha(0.5)

    # out_fig_file = output_dir + '/' + window_title + '.pdf'
    # plt.savefig(out_fig_file)
    out_fig_file = g_output_dir + '/' + title + '.pdf'
    plt.savefig(out_fig_file)

    # plt.show(block=False)

########################################################################################################################



def plot_scatter(error_data, title, desc):
    colors = ['r', 'b', 'g', 'c', 'm', 'y', '0.75']
    plt.figure()
    plt.title(desc)
    fig = plt.gcf()
    fig.canvas.set_window_title(desc)
    plt.xlabel('X')
    plt.ylabel('Y')

    ax1 = fig.add_subplot(111)
    alpha = 0.2

    groups = ['jaw','nose','mouth','left_eye','right_eye','left_brow','right_brow']

    f = open(g_output_dir + '/' + title + '_' + desc + '.txt','wt')
    for i in range(len(groups)):
        name = groups[i]
        x, y = zip(*error_data['error_table'][name + '_error_vec'])
        ax1.scatter(x,y, c=colors[i], label=name, alpha=alpha)
        f.write('%s\tmean = %.3f, std = %.3f\n'%(name, np.mean(x), np.std(x)))
    f.close()
    # out_fig_file = output_dir + '/' + window_title + '.pdf'
    # plt.savefig(out_fig_file)
    leg = plt.legend(loc=0, fontsize='medium')
    leg.get_frame().set_alpha(0.5)

    out_fig_file = g_output_dir + '/' + title + '_' + desc + '.pdf'
    plt.savefig(out_fig_file)
    # plt.show(block=True)

########################################################################################################################

def get_files_under_dir(dir, estimated_error_threshold = 10000):
    output = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('pts'):
                file_rel_path = relpath(os.path.join(root, name), dir)
                estimated_error_file_path = '/'.join((root, file_rel_path))[:-3] + 'txt'
                if os.path.isfile(estimated_error_file_path):
                    with open(estimated_error_file_path, 'r') as f:
                        estimated_error = float(f.readline())
                        if estimated_error > estimated_error_threshold:
                            continue

                output.append(file_rel_path)

    return output
########################################################################################################################


def results_evaluation(json_params):

    # get common files
    first_json_record = json_params['Data'][0]
    intersect_data = json_params['intersect_data']
    recall_rate = json_params['recall_rate']
    first_estimated_files_root = first_json_record['estimated_dir']
    first_gt_files_root = first_json_record['gt_dir']
    gt_files_list = get_files_under_dir(first_gt_files_root)
    # first_file_list = [str(f) for f in os.listdir(first_json_record ['estimated_dir']) if f.endswith('pts')]
    first_file_list = get_files_under_dir(first_estimated_files_root)
    global g_common_filenames
    g_common_filenames = set(first_file_list)
    g_common_filenames = g_common_filenames.intersection(set(gt_files_list))
    for data_dir in json_params['Data']:
        # filenames = [str(f) for f in os.listdir(data_dir['estimated_dir']) if f.endswith('pts')]
        # filenames = get_files_under_dir(data_dir['estimated_dir'], 0.035)
        filenames = get_files_under_dir(data_dir['estimated_dir'], 10000)
        g_common_filenames = g_common_filenames.intersection(set(filenames))

    print('common filenames: ' + str(len(g_common_filenames)))
    error_data_per_dir = []
    step_loss_error_data_per_dir = []
    for data_dir in json_params['Data']:
        filenames = get_files_under_dir(data_dir['estimated_dir'], 10000)
        filenames = set(filenames).intersection(gt_files_list)
        if intersect_data:
            filenames = g_common_filenames
        error_data_per_dir.append(compute_error_data(data_dir['estimated_dir'],
                                                     data_dir['gt_dir'],
                                                     data_dir['rect_dir'],
                                                     data_dir['image_dir'],
                                                     data_dir['desc'],
                                                     data_dir['flm_count'],
                                                     filenames=filenames,
                                                     recall_rate=recall_rate),
                                                        )
        #plot_scatter(error_data_per_dir[-1], 'Error Scatter Plot', data_dir['desc'])

        # step_loss_error_data_per_dir.append(compute_error_data(data_dir['estimated_dir'],
        #                                              data_dir['gt_dir'],
        #                                              data_dir['rect_dir'],
        #                                              data_dir['desc'],
        #                                              True,
        #                                              0.03))
        #plot_scatter(step_loss_error_data_per_dir[-1], 'Step Loss Error Scatter Plot', data_dir['desc'])

    print_statistics(error_data_per_dir)
    plot_ced(error_data_per_dir, 'Precision vs Recall', 'ced_recall_vs_precision')
    plot_ced(error_data_per_dir, 'Estimated Regression Error', 'ced_estimated_error')
    plot_ced(error_data_per_dir, 'Normalized Point-to-Point Error All-points','ced_data_all')
    plot_ced(error_data_per_dir, 'Normalized Point-to-Point Error Non-Jaw-points',
             'ced_non_jaw')
    plot_bar(error_data_per_dir, 'CED at 0.08', 0.08)
    plot_bar(error_data_per_dir, 'CED at 0.05', 0.05)

    #plot_qualitative_results()




########################################################################################################################

if __name__ == "__main__":

    with open('evaluation.json') as f:
        data = json.load(f)

    g_output_dir = data['output_dir']
    g_dump_files = data['dump_output_files']
    g_metric = data['metric']
    g_normalization = data['normalization']
    # files = data['all_filenames']
    # with open(files,'rt') as f:
    #     g_all_filenames = set([n.strip() for n in f.readlines()])
    # g_common_filenames = g_all_filenames.copy()

    results_evaluation(data)
   # cProfile.run('results_evaluation(data)')

