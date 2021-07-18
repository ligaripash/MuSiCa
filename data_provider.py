from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
import PIL
import common_params
from pathlib import Path
import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import tensorflow as tf
import detect
import utils
import os
from image_preprocessing import preprocess_image_menpo, transform_image_space_to_norm_space
import imutils
import cv2
# from mdm_train_coarse_to_fine import build_image_pyr

FLAGS = tf.app.flags.FLAGS
# BB_SIZE = 142.0
# BB_SIZE = 128.0
# gil - change for image pyrmaid
# BB_SIZE = 256.0
# The size the initialization points were normalized with
# INITIAL_POINTS_BB_SIZE = 142.0

NEW_CROP_METHOD = 1

def build_reference_shape_per_pose_by_files(filenames, data_dir, bb_dir, pose_dir, num_of_poses=3):

    ref_shapes = {}
    landmarks = [[] for i in range(num_of_poses)]

    print('Building reference shapes from %d files'%len(filenames))
    # collect lms by pose
    i=0
    pose=0
    for file in filenames:
        if i % 1000 == 0:
            print(i)
        i += 1
        try:
            gt = mio.import_landmark_file(data_dir+file[:-3]+'pts')
            bb = mio.import_landmark_file(bb_dir + file[:-3]+'pts')
        except:
            print data_dir+file[:-3]+'pts not found'
            continue
        if not (bb.lms.points[2][0] - bb.lms.points[0][0] > 10 and bb.lms.points[2][1] - bb.lms.points[0][1] > 10):
                    continue

        if pose_dir != '':
            try:
                with open(pose_dir + file[:-3] + 'txt','rt') as f:
                    pose = int(next(f))
                    if not pose in range(num_of_poses):
                        continue
            except:
                continue
        # transform lms into BB - this code changes proportions
        #x_s = BB_SIZE /(bb.lms.points[2, 1] - bb.lms.points[0, 1])
        #y_s = BB_SIZE / (bb.lms.points[2, 0] - bb.lms.points[0, 0])
        ###### code for rectangle bb ######
        xp = bb.lms.points[2][1] - bb.lms.points[0][1]
        yp = bb.lms.points[2][0] - bb.lms.points[0][0]
        x_s, y_s = 0,0
        if xp > yp:
            x_s = common_params.BB_SIZE / xp
            y_s = x_s * np.max([1, (xp / 4) / yp])
        else:
            y_s = common_params.BB_SIZE / yp
            x_s = y_s * np.max([1, (yp / 4) / xp])
        x_t = bb.lms.points[0, 1]
        y_t = bb.lms.points[0, 0]
        gt.lms.points[:, 1] = (gt.lms.points[:, 1] - x_t) * x_s
        gt.lms.points[:, 0] = (gt.lms.points[:, 0] - y_t) * y_s

        landmarks[pose] += [gt.lms]
    # generate refs by pose
    for pose in range(num_of_poses):
        print('Pose %d: %d files'%(pose, len(landmarks[pose])))
        ref_shapes[str(pose)] = PointCloud(compute_reference_shape(landmarks[pose], False).points.astype(np.float32))

    if num_of_poses == 1:
        return ref_shapes['0']
    else:
        return ref_shapes

#%#%#%#%#%#%#%#%#%#%#%#%#
def build_reference_shape_by_yaw(filenames, data_dir, bb_dir, pose_dir, roll_dir, yaw_range):

    ref_shapes = {}
    landmarks = {}
    for f, t in yaw_range:
        landmarks['%d_%d'%(f, t)] = []

    print('Building reference shapes from %d files'%len(filenames))
    # collect lms by pose
    i=0
    for file in filenames:
        if i % 1000 == 0:
            print(i)
        i += 1
        gt = mio.import_landmark_file(data_dir+file[:-3]+'pts')
        bb = mio.import_landmark_file(bb_dir + file[:-3]+'pts')
        try:
            with open(pose_dir + file[:-3] + 'txt','rt') as f:
                pose = float(next(f))
                if not pose >= -90 and pose <= 90:
                    continue
            with open(roll_dir + file[:-3] + 'txt', 'rt') as f:
                roll = float(next(f))
        except:
            continue

        ###### code for rectangle bb ######
        # roll
        bb_center = (bb.lms.points[0,:] +  bb.lms.points[2,:])*0.5
        gt.lms.points = rotate_points(gt.lms.points, bb_center, -roll)
        gt.lms.points = scale_lms(bb, gt.lms.points)

        yaw_key = get_pose_key(pose, yaw_range, with_out_of_range=False)
        if not(yaw_key == ''):
            landmarks[yaw_key].append(gt.lms)
    # generate refs by pose
    for yaw_key, lms in landmarks.items():
        print('Pose %s: %d files'%(yaw_key, len(landmarks[yaw_key])))
        ref_shapes[yaw_key] = PointCloud(compute_reference_shape(landmarks[yaw_key], False).points.astype(np.float32))

    return ref_shapes
#%#%#%#%#%#%#%#%#%#%#%#%#

def read_pose(name, yaw_dir, pitch_dir, roll_dir):
    pose_name = name[:-3] + 'txt'
    try:
        with open(yaw_dir + pose_name, 'rt') as f:
            yaw = float(next(f))
        with open(pitch_dir + pose_name) as f:
            pitch = float(next(f))
        with open(roll_dir + pose_name) as f:
            roll = float(next(f))
    except:
        return 0,0,0, False

    return yaw, pitch, roll, True

#%#%#%#%#%#%#%#%#%#%#%#%#

def build_reference_shape_by_pose(filenames, data_dir, bb_dir, yaw_dir, pitch_dir, roll_dir, yaw_range, pitch_range):

    ref_shapes = {}
    landmarks = {}
    for y_f, y_t in yaw_range:
        for p_f, p_t in pitch_range:
            landmarks['y_%d_%d_p_%d_%d'%(y_f, y_t, p_f, p_t)] = []

    print('Building reference shapes from %d files'%len(filenames))
    # collect lms by pose
    i=0
    for file in filenames:
        if i % 1000 == 0:
            print(i)
        i += 1
        gt = mio.import_landmark_file(data_dir+file[:-3]+'pts')
        bb = mio.import_landmark_file(bb_dir + file[:-3]+'pts')

        yaw, pitch, roll, _ = read_pose(file, yaw_dir, pitch_dir, roll_dir)

        bb_center = (bb.lms.points[0, :] + bb.lms.points[2, :]) * 0.5
        gt.lms.points = rotate_points(gt.lms.points, bb_center, -roll)
        gt.lms.points = scale_lms(bb, gt.lms.points)

        pose_key = get_yaw_pitch_key(yaw, pitch, yaw_range, pitch_range, with_out_of_range=False)
        if not(pose_key == ''):
            # roll first
            landmarks[pose_key].append(gt.lms)
    # generate refs by pose
    for pose_key, lms in landmarks.items():
        print('Pose %s: %d files'%(pose_key, len(landmarks[pose_key])))
        ref_shapes[pose_key] = PointCloud(compute_reference_shape(landmarks[pose_key], False).points.astype(np.float32))

    return ref_shapes

#%#%#%#%#%#%#%#%#%#%#%#%#
def scale_lms(bb, points):
    w = bb.lms.points[2][1] - bb.lms.points[0][1]
    h = bb.lms.points[2][0] - bb.lms.points[0][0]
    x_s, y_s = 0, 0
    if w > h:
        x_s = common_params.BB_SIZE / w
        y_s = x_s * np.max([1, (w / 4) / h])
    else:
        y_s = common_params.BB_SIZE / h
        x_s = y_s * np.max([1, (h / 4) / w])

    x_t = bb.lms.points[0, 1]
    y_t = bb.lms.points[0, 0]
    points[:, 1] = (points[:, 1] - x_t) * x_s
    points[:, 0] = (points[:, 0] - y_t) * y_s

    return points

# assumption: yaw_range is sorted
# return key + bin index (for eval)
# assumption: yaw_range is sorted
# return key + bin index (for eval)
def get_pose_key(pose, pose_range, with_out_of_range = True):

    if pose < pose_range[0][0] or pose > pose_range[-1][1]:
        if not(with_out_of_range):
            print('Out of range pose: ' + str(pose))
            return ''
        if pose < pose_range[0][0]:
            return '%d_%d'%tuple(pose_range[0])
        # pose > yaw_range[-1][1]
        return '%d_%d' % tuple(pose_range[-1])

    # pose within range
    for f, t in pose_range:
        if pose <= t:
            return '%d_%d'%(f, t)

def get_pose_ind(pose, pose_range, with_out_of_range = True):

    if pose < pose_range[0][0] or pose > pose_range[-1][1]:
        if not(with_out_of_range):
            print('Out of range pose: ' + str(pose))
            return -1
        if pose < pose_range[0][0]:
            return 0
        # pose > yaw_range[-1][1]
        return len(pose_range)-1

    # pose within range
    ind = 0
    for f, t in pose_range:
        if pose <= t:
            return ind
        ind += 1

def get_yaw_pitch_key(yaw, pitch, yaw_range, pitch_range, with_out_of_range = True):

    yaw_key = get_pose_key(yaw, yaw_range, with_out_of_range)
    if not(pitch_range):
        return yaw_key

    pitch_key = get_pose_key(pitch, pitch_range, with_out_of_range)
    return 'y_' + yaw_key + '_p_' + pitch_key

def get_yaw_pitch_ind(yaw, pitch, yaw_range, pitch_range, with_out_of_range = True):
    yaw_ind = get_pose_ind(yaw, yaw_range, with_out_of_range)
    pitch_ind = get_pose_ind(pitch, pitch_range, with_out_of_range)

    # for with_out_of_range = False : negative ind means out of range
    if not(with_out_of_range) and (yaw_ind < 0 or pitch_ind < 0):
        return -1

    return yaw_ind * 3 + pitch_ind

def create_pose_range(angles):
        return [(f, t) for (f, t) in zip(angles[0:-1], angles[1:])]

##################################################################################################################

# gil

def rgb_to_grey(im):
    """Converts menpo Image to grey if color"""
    assert im.n_channels in [1, 3]
    if im.n_channels == 1:
        return im



def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale

    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im


def align_reference_shape(reference_shape, bb):
    min_xy = tf.reduce_min(reference_shape, 0)
    max_xy = tf.reduce_max(reference_shape, 0)
    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    # reference_shape_bb = tf.pack([[min_x, min_y], [max_x, min_y],
    #                              [max_x, max_y], [min_x, max_y]])
    reference_shape_bb = [[min_x, min_y], [max_x, min_y],
                          [max_x, max_y], [min_x, max_y]]

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    return tf.add(
        (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio,
        tf.reduce_mean(bb, 0),
        name='initial_shape')


def random_shape(gts, reference_shape, pca_model):
    """Generates a new shape estimate given the ground truth shape.

    Args:
      gts: a numpy array [num_landmarks, 2]
      reference_shape: a Tensor of dimensions [num_landmarks, 2]
      pca_model: A PCAModel that generates shapes.
    Returns:
      The aligned shape, as a Tensor [num_landmarks, 2].
    """

    def synthesize(lms):
        return detect.synthesize_detection(pca_model, menpo.shape.PointCloud(
            lms).bounding_box()).points.astype(np.float32)

    bb, = tf.py_func(synthesize, [gts], [tf.float32])
    shape = align_reference_shape(reference_shape, bb)
    shape.set_shape(reference_shape.get_shape())

    return shape


def get_noisy_init_from_bb(reference_shape, bb, noise_percentage=.02):
    """Roughly aligns a reference shape to a bounding box.

    This adds some uniform noise for translation and scale to the
    aligned shape.

    Args:
      reference_shape: a numpy array [num_landmarks, 2]
      bb: bounding box, a numpy array [4, ]
      noise_percentage: noise presentation to add.
    Returns:
      The aligned shape, as a numpy array [num_landmarks, 2]
    """
    bb = PointCloud(bb)
    reference_shape = PointCloud(reference_shape)

    bb = noisy_shape_from_bounding_box(
        reference_shape,
        bb,
        noise_percentage=[noise_percentage, 0, noise_percentage]).bounding_box(
    )

    return align_shape_with_bounding_box(reference_shape, bb).points


def rescale_im(im, ref_shape, margin = 0.3):

    # calc bb dimensions
    xp = im.landmarks['bb'].lms.points[2][1] - im.landmarks['bb'].lms.points[0][1]
    yp = im.landmarks['bb'].lms.points[2][0] - im.landmarks['bb'].lms.points[0][0]
    # find bb scale in comparison to 142 (longer dimension)
    if xp > yp:
        xs = common_params.BB_SIZE / xp
        ys = xs * np.max([1, (xp / 4) / yp])
    else:
        ys = common_params.BB_SIZE / yp
        xs = ys * np.max([1, (yp / 4) / xp])
    # left upper bb corner
    xt = im.landmarks['bb'].lms.points[0][1]
    yt = im.landmarks['bb'].lms.points[0][0]
    # calc reference model coordinates on the original image
    for i in range(ref_shape.points.shape[0]):
        # gil - scale the init points from their original scale to the new BB_SIZE
        xss = common_params.INITIAL_POINTS_BB_SIZE / common_params.BB_SIZE
        yss = common_params.INITIAL_POINTS_BB_SIZE / common_params.BB_SIZE
        ref_shape.points[i][1] = ref_shape.points[i][1] / xss
        ref_shape.points[i][0] = ref_shape.points[i][0] / yss
        ref_shape.points[i][1] = ref_shape.points[i][1] / xs + xt
        ref_shape.points[i][0] = ref_shape.points[i][0] / ys + yt
    im.landmarks['__initial'] = ref_shape
    ##### new code - margin per dim
    x_m = margin*xp
    y_m = margin*yp
    min_indices = [yt - y_m, xt - x_m]
    max_indices = [im.landmarks['bb'].lms.points[2][0] + y_m, im.landmarks['bb'].lms.points[2][1] + x_m]
    #bbs_plus_margin_input_image = [min_indices[1], min_indices[0], max_indices[1], max_indices[0]]

    # the margin is set by the maximal dimension
    im = im.crop(min_indices, max_indices, constrain_to_boundary=True, return_transform=False)
    im = im.rescale((xs, ys))

    return im #, (xs, ys), bbs_plus_margin_input_image

def load_image_for_train_hfd(name, data_dir, gt_root, pose_root, bb_root, reference_shapes, margin = 0.3):
    # print(name)
    try:
        im = mio.import_image(data_dir + name)

    except:
        print 'problem with ' + data_dir + name

    # print('reading gt file', gt_root + name[:-3] + 'pts')
    im.landmarks['PTS'] = mio.import_landmark_file(gt_root + name[:-3] + 'pts')
    group = im.landmarks.group_labels[0]
    im.landmarks['bb'] = mio.import_landmark_file(bb_root + name[:-3] + 'pts')

    # gil try using the same frontal initialization for all faces
    if common_params.FRONTAL_POSE_ONLY:
        pose = 0
        reference_shape = reference_shapes
    else:
        with open(pose_root + name[:-3] + 'txt') as f:
            pose = int(next(f))

        # gil - vra pose 3 is roll - for now init with 0
        if pose == 3:
            pose = 0
        reference_shape = reference_shapes[str(pose)]


    # image mainpulations
    im = rescale_im(im, reference_shape.copy(), margin)
    # make sure image is uni-size for all images
    pSize = np.ceil(common_params.BB_SIZE * (1 + 2 * margin))
    im = im.resize((pSize, pSize))

    im = grey_to_rgb(im)

    shape = im.landmarks[group].lms
    init = im.landmarks['__initial'].lms
    bb = im.landmarks['bb'].lms
    pixels = im.pixels.transpose(1, 2, 0)
    # gil
    if False:
        pSize = int(np.ceil(common_params.BB_SIZE * (1 + 2 * margin)) + 3)
        padded_shape = [pSize, pSize, pixels.shape[2]]
        padded_im = np.random.rand(*padded_shape).astype(np.float32)

        height, width = im.shape[:2]
        dy = max(int(np.ceil(common_params.BB_SIZE * margin) - bb.points[0][0]), 0)
        dx = max(int(np.ceil(common_params.BB_SIZE * margin) - bb.points[0][1]), 0)
        add_delta(shape, dy, dx)
        add_delta(init, dy, dx)
        add_delta(bb, dy, dx)
        padded_im[dy:(height + dy), dx:(width + dx)] = pixels

    # return padded_im, shape, init, bb
    # gil
    return pixels, shape, init, bb

def get_hfd_image_shape(margin = 0.3, channel_count=3, pyr_level=0):
    pSize = int(np.ceil(common_params.BB_SIZE * (1 + 2 * margin)) + 3)
    pSize = np.ceil(pSize / (2 ** pyr_level) ) + 1
    # return [pSize, pSize, 3]
    # gil
    return [pSize, pSize, channel_count]


########################################################################################################################

def add_bb_noise(bb):
    """
    Enlarge bb by scaling it
    :param bb:
    :return:
    """

    center = bb.centre()
    center_4_times = np.array(list(center) * 4)
    bb_v = bb.as_vector()
    bb_center = bb_v - center_4_times
    scale_factor_x = np.random.uniform(1, 1.1)
    scale_factor_y = np.random.uniform(1, 1.2)
    scale_vector = np.array([scale_factor_x, scale_factor_y] * 4)
    scaled_bb_center = np.multiply(bb_center, scale_vector)
    bb_width = bb_v[5] - bb_v[1]
    bb_height = bb_v[2] - bb_v[0]
    #translation up to translation_factor of the new width and height
    translation_factor = 0.2

    t_y = np.random.uniform(-(translation_factor * bb_height), translation_factor * bb_height)
    t_x = np.random.uniform(-(translation_factor * bb_width), translation_factor * bb_width)
    t = np.array([t_x, t_y] * 4)
    scaled_bb_center = scaled_bb_center + t
    back_to_image = scaled_bb_center + center_4_times
    return back_to_image

########################################################################################################################


def add_color_jetting(t_im):

    eta = 0.4
    eta_min_channel_0 = np.random.uniform(0, eta)
    eta_min_channel_1 = np.random.uniform(0, eta)
    eta_min_channel_2 = np.random.uniform(0, eta)

    eta_max_channel_0 = np.random.uniform(1 - eta, 1)
    eta_max_channel_1 = np.random.uniform(1 - eta, 1)
    eta_max_channel_2 = np.random.uniform(1 - eta, 1)

    pix_array = t_im.pixels
    pix_array[0,pix_array[0,] < eta_min_channel_0] = 0
    pix_array[0, pix_array[0,] > eta_max_channel_0] = 1

    pix_array[1,pix_array[1,] < eta_min_channel_1] = 0
    pix_array[1, pix_array[1,] > eta_max_channel_1] = 1

    pix_array[2,pix_array[2,] < eta_min_channel_2] = 0
    pix_array[2, pix_array[2,] > eta_max_channel_2] = 1
    min_0 = pix_array[0,].min()
    max_0 = pix_array[0,].max()
    min_1 = pix_array[1,].min()
    max_1 = pix_array[1,].max()
    min_2 = pix_array[2,].min()
    max_2 = pix_array[2,].max()


    pix_array[0,] = (pix_array[0,] - min_0) / max_0
    pix_array[1,] = (pix_array[1,] - min_1) / max_1
    pix_array[2,] = (pix_array[2,] - min_2) / max_2

    t_im.pixels = pix_array
    return t_im

########################################################################################################################

def add_occlusion(im):
    gamma = 0.5
    #choose a center pixel
    im_height = float(im.shape[0])
    im_width = float(im.shape[1])

    center_x = int(np.random.uniform(0,im_width))
    center_y = int(np.random.uniform(0, im_height))

    patch_width =  int(np.random.uniform(1, gamma * im_width))
    patch_height = int(np.random.uniform(1, gamma * im_height))

    ul = (center_x - patch_width//2, center_y - patch_height//2)

    patch = np.random.rand(patch_height, patch_width, 3)
    patch_img = PIL.Image.fromarray(patch, 'RGB')
    pil_image = im.as_PILImage()
    pil_image.paste(patch_img , ul)

    im.pixels = np.array(np.transpose(np.array(pil_image), (2,0,1)),dtype=np.float32) / 255
    return im




########################################################################################################################

def augment_image(im):
    """
    To simulate extreme poses that do not exists on the training set, we rescale the image
    Without preserving aspect ratio.
    Later we enlarge the bounding box to simulate FD detection inaccuracies.
    :param im:
    :return: transformed menpo image with GT and FD bounding box
    """
    # First crop out the face to save reduce computation load
    bb = im.landmarks['bb'].lms
    bb_vec = bb.as_vector()
    bb_ul = (np.array([bb_vec[0], bb_vec[1]]) - bb.centre()) * 2
    bb_lr = (np.array([bb_vec[4], bb_vec[5]]) - bb.centre()) * 2
    ul = bb_ul + bb.centre()
    lr = bb_lr + bb.centre()
    im = im.crop(ul, lr, constrain_to_boundary=True)
    if im.pixels.shape[0] == 1:
        pix = np.zeros((3, im.pixels.shape[1], im.pixels.shape[2]))
        pix[:,] = im.pixels
        im.pixels = pix

    beta = 0.3
    cx = np.random.uniform(-beta, beta)
    cy = np.random.uniform(-beta, beta)
    fx = 1.0
    fy = np.random.uniform(0.6, 1.4)
    max_rotation = 30
    theta = np.random.uniform(-max_rotation, max_rotation)

    rotation = menpo.transform.Rotation.init_from_2d_ccw_angle(theta)
    shear = menpo.transform.Affine(np.array([[1, cx, 0],[cy, 1, 0], [0,0,1]]))
    scale = menpo.transform.Affine(np.array([[fx, 0, 0],[0, fy, 0], [0,0,1]]))
    T = scale.compose_after(shear).compose_after(rotation)

    t_im = im.transform_about_centre(T)

    t_im = add_color_jetting(t_im)
    t_im = add_occlusion(t_im)


    new_bb = t_im.landmarks['PTS'].lms.bounding_box()

    #new_bb contains the gt bounding box
    augmented_bb = add_bb_noise(new_bb)
    augmented_bb = augmented_bb.reshape((4,2))
    augmented_bb = menpo.shape.PointCloud(augmented_bb)
    t_im.landmarks['bb'] = menpo.landmark.LandmarkGroup.init_with_all_label(augmented_bb)

    return t_im


def rgb2gray(rgb):

    pixels = rgb.pixels.transpose(1,2,0).astype('float32')
    single_channel = np.dot(pixels, [0.2989, 0.5870, 0.1140])
    single_channel = np.expand_dims(single_channel, 0)
    three_channels = np.vstack([single_channel]*3)
    rgb.pixels = three_channels

    return rgb
########################################################################################################################
def load_image_for_train_new(name, image_dir, gt_dir, bb_root, reference_shape, margin, roll_support = False):

    im = mio.import_image(image_dir+name)
    im.landmarks['PTS'] = mio.import_landmark_file(gt_dir + name[:-3] + 'pts')
    #group = im.landmarks.group_labels[0]
    im.landmarks['bb'] = mio.import_landmark_file(bb_root + name[:-3] + 'pts')

    # Gil - experiment with new augmentation: instead of manipulating the init points, manipulate the image
    # import random
    if np.random.rand() < 0.5:
        im = augment_image(im)

    im = rescale_im(im, reference_shape.copy(), margin)
    im = grey_to_rgb(im)

    if np.random.rand() < 0.5:
        im = rgb2gray(im)
        
    im = menpo.feature.normalize_std(im)

    pixels = im.pixels.transpose(1, 2, 0)
    shape = im.landmarks['PTS'].lms
    init = im.landmarks['__initial'].lms
    bb = im.landmarks['bb'].lms
    # gil - we use automatic padding
    return pixels, shape, init, bb
    # rgb_margin_rotated, T, rot_translate2bb_T, translate2bb_T =  transform_image_space_to_norm_space(np.uint8(im), bbs_plus_margin_input_image, roll)
    # shape.points = transform_points(shape.points, rot_translate2bb_T) # rotate and translate GT
    # init.points = transform_points(shape.points, translate2bb_T)
#    bb.points =
    #pixels = im.pixels.transpose(1, 2, 0)
    ############## NFD Remove Padding ##############
    # padded_shape = get_image_shape(margin)
    # padded_im = np.random.rand(*padded_shape).astype(np.float32)
    #
    # height, width = im.shape[:2]
    # # centralize image - needed for augmentation
    # #dy = max(int(np.ceil(BB_SIZE * margin_up) - bb.points[0][0]), 0)
    # #dx = max(int(np.ceil(BB_SIZE * margin_x) - bb.points[0][1]), 0)
    # dy = int(np.ceil((padded_shape[0] - height) / 2))
    # dx = int(np.ceil((padded_shape[1] - width) / 2))
    # add_delta(shape, dy, dx)
    # add_delta(init, dy, dx)
    # add_delta(bb, dy, dx)
    #
    # padded_im[dy:(height + dy), dx:(width + dx)] = pixels
    # ################################################
    # return padded_im, shape, init, bb

def load_image_for_train(name, data_dir, gt_dir, bb_root, yaw_root, pitch_root, roll_root, reference_shapes, yaw_range,
                         pitch_range, margin_ratio_x = 0.3, margin_ratio_down = 0.3, margin_ratio_up = 0.3, roll_support = False):
    #print(name)
    try:
        im = mio.import_image(data_dir+name)
    except:
        raise Exception('Error in mio.import_image with file ' + data_dir + name)

    try:
        im.landmarks['PTS'] = mio.import_landmark_file(gt_dir + name[:-3] + 'pts')
    except:
        raise Exception('Error in mio.import_landmark_file with file ' + gt_dir + name[:-3] + 'pts')
    #group = im.landmarks.group_labels[0]
    try:
        im.landmarks['bb'] = mio.import_landmark_file(bb_root +  name[:-3] + 'pts')
    except:
        raise Exception('Error in mio.import_landmark_file with file ' + gt_dir + name[:-3] + 'pts')


    # yaw - for initialization
    with open(yaw_root + name[:-3] + 'txt') as f:
        yaw = float(next(f))
    with open(pitch_root + name[:-3] + 'txt') as f:
        pitch = float(next(f))
    reference_shape = reference_shapes[get_yaw_pitch_key(yaw, pitch, yaw_range, pitch_range)]

    # for evaluation
    #bbs = im.landmarks['bb'].lms
    #h_im, w_im = im.shape
    #bbs_m, bbs_n, bbs_output = menpo.get_crop_indexes(bbs, w_im, h_im, BB_SIZE, MARGIN_X, MARGIN_UP, MARGIN_DOWN)
    # image mainpulations
    im, scale, bbs_plus_margin_input_image = rescale_im(im, reference_shape.copy(), margin_ratio_x, margin_ratio_down, margin_ratio_up)
    im = grey_to_rgb(im)
    pixels = im.pixels.transpose(1, 2, 0)
    shape = im.landmarks['PTS'].lms
    init = im.landmarks['__initial'].lms
    bb = im.landmarks['bb'].lms

    # deal with rotation + create a transformation matrix
    with open(roll_root + name[:-3] + 'txt') as f:
        roll = float(next(f))

    # T is the concatenated transformation matrix of all transformation from the original image:
    # (1) translate to cropped image, (2) scale to BB_SIZE, (3) rotate by -roll, (4) translate to rotated bb
    # added_T is the concatenation of only (3)-(4) which are the actual transformations in this function.
    # This matrix should be applied on the initial points ((1) and (2) were applied on bb/__initial/shape landmarks
    # in rescale_im function)
    # rot_translate2bb_T - only (4), for the initial points that are translated but not rotated
    rgb_margin_rotated, T, rot_translate2bb_T, translate2bb_T =  transform_image_space_to_norm_space(np.uint8(im), bbs_plus_margin_input_image, roll)
    shape.points = transform_points(shape.points, rot_translate2bb_T) # rotate and translate GT
    init.points = transform_points(shape.points, translate2bb_T)
#    bb.points =
    #pixels = im.pixels.transpose(1, 2, 0)
    ############## NFD Remove Padding ##############
    padded_shape = get_image_shape(margin_ratio_x, margin_ratio_down, margin_ratio_up)
    padded_im = np.random.rand(*padded_shape).astype(np.float32)

    height, width = im.shape[:2]
    # centralize image - needed for augmentation
    #dy = max(int(np.ceil(BB_SIZE * margin_up) - bb.points[0][0]), 0)
    #dx = max(int(np.ceil(BB_SIZE * margin_x) - bb.points[0][1]), 0)
    dy = int(np.ceil((padded_shape[0] - height) / 2))
    dx = int(np.ceil((padded_shape[1] - width) / 2))
    add_delta(shape, dy, dx)
    add_delta(init, dy, dx)
    add_delta(bb, dy, dx)

    padded_im[dy:(height + dy), dx:(width + dx)] = pixels
    ################################################
    return padded_im, shape, init, bb

def rotate_image(image, center, angle):
    row,col = image.shape[:2]
    if not(center):
        center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate_points(pts, center, roll):
    points_in_image_center_system = np.array([(x, y, 1 ) for (y,x) in pts])
    inv_points_mat = np.transpose(points_in_image_center_system)
    rot_mat = cv2.getRotationMatrix2D(tuple(center), roll, 1)
    rotated_points = rot_mat.dot(inv_points_mat)

    # final_points = np.transpose(np.add(rotated_points, b))
    final_points = np.transpose(rotated_points)
    swap = np.array([(x, y) for (y, x) in final_points])
    return swap

def transform_points(points, T):
    """ Apply transformation on the points collection """

    homo_points = np.array([(x, y, 1) for (y, x) in points])
    t_points = np.array([T.dot(v) for v in homo_points ])
    swap = np.array([(x,y) for (y,x,z) in t_points])
    return swap

def get_image_shape(margin):
    xSize = int(np.ceil(common_params.BB_SIZE * (1 + 2*margin)) + 3)
    ySize = int(np.ceil(common_params.BB_SIZE * (1 + 2*margin)) + 3)

    return [ySize, xSize, 3]


def add_delta(p, dy, dx):
    p.points[:,0] += dy
    p.points[:, 1] += dx
    return p

def load_images_debug(im):
    im_to_write = im.copy()
    lm1 = im.landmarks['__initial'].lms.points
    lm2 = im.landmarks['PTS'].lms.points
    lm3 = bb.lms.points
    for li in range(lm1.shape[0]):
        im_to_write.pixels[:, int(lm1[li, 0]), int(lm1[li, 1])] = [1, 0, 0]
        im_to_write.pixels[:, int(lm2[li, 0]), int(lm2[li, 1])] = [0, 1, 0]
    for li in range(int(lm3[2][0] - lm3[0][0])):
        im_to_write.pixels[:, int(lm3[0][0]) + li, int(lm3[0][1])] = [0, 0, 1]
        im_to_write.pixels[:, int(lm3[0][0]) + li, int(lm3[2][1])] = [0, 0, 1]
    for li in range(int(lm3[2][1] - lm3[0][1])):
        im_to_write.pixels[:, int(lm3[0][0]), int(lm3[0][1]) + li] = [0, 0, 1]
        im_to_write.pixels[:, int(lm3[2][0]), int(lm3[0][1]) + li] = [0, 0, 1]
    from menpo.io.output.base import export_image
    debug_path = './debug/'
    export_image(im_to_write, str(debug_path + im_to_write.path.stem + '.png'), overwrite=True)


def save_reference_shape(reference_shape, pose, path, group = None):
    if not os.path.isdir(path):
        os.mkdir(path)
    if pose == 0:
        mio.export_pickle(reference_shape.points, path + 'reference_shape.pkl', overwrite=True)
    else:
        for i in range(len(reference_shape)):
            mio.export_pickle(reference_shape[str(i)].points, path + 'reference_shape_%d.pkl'%i, overwrite=True)
    print('created reference_shape.pkl')

def load_reference_shape(path, pose):
    if pose == 0:
        reference_shape = PointCloud(mio.import_pickle(path + 'reference_shape.pkl'))
    else:
        reference_shape = {}
        for i in range(3):
            reference_shape[str(i)] = PointCloud(mio.import_pickle(path + 'reference_shape_%d.pkl' % i))

    print('loaded reference_shape')
    return reference_shape

def load_reference_shape_w_yaw(path, yaw_range):
    reference_shape = {}
    for (f,t) in yaw_range:
        key = '%d_%d'%(f, t)
        reference_shape[key] = PointCloud(mio.import_pickle(path + 'reference_shape_%s.pkl' % key))

    print('loaded reference_shape')
    return reference_shape

def load_reference_shape_w_pose(path, pose_range):
    reference_shape = {}
    for pose in pose_range:
        key = 'y_%d_%d_p_%d_%d' % pose
        reference_shape[key] = PointCloud(mio.import_pickle(path + 'reference_shape_%s.pkl' % key))

    print('loaded reference_shape')
    return reference_shape


def save_reference_shape_w_pose(reference_shape, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    for key in reference_shape.keys():
            print('export reference_shape_%s.pkl'%key)
            mio.export_pickle(reference_shape[key].points, path + 'reference_shape_%s.pkl'%key, overwrite=True)
    print('created reference_shape.pkl')

def is_file_ok(name, gt_dir, bb_root, pose_root, patch_size = 14, margin = 0.3):
    pts = mio.import_landmark_file(gt_dir + name[:-3] + 'pts')
    try:
        bb = mio.import_landmark_file(bb_root + name[:-3] + 'pts')
    except:
        return False

    if not (bb.lms.points[2][0] - bb.lms.points[0][0] > 10 and
                        bb.lms.points[2][1] - bb.lms.points[0][1] > 10):
        return False

    marg = margin - patch_size / common_params.BB_SIZE / 2
    if not (bb.lms.points[0][0] - pts.lms.bounding_box().points[0][0] < marg *
        bb.lms.range()[0] and
                        pts.lms.bounding_box().points[0][1] - bb.lms.points[0][1] < marg * bb.lms.range()[1] and
                        bb.lms.points[2][0] - pts.lms.bounding_box().points[2][0] < marg * bb.lms.range()[0] and
                        pts.lms.bounding_box().points[2][1] - bb.lms.points[2][1] < marg * bb.lms.range()[1]):
        return False

    try:
        with open(pose_root + name[:-3] + 'txt') as f:
            pose = int(next(f))
            if pose < 0 or pose > 2:
                return False
    except:
        return False

    return True

def is_nfd_file_ok(name, gt_dir, bb_root, pose_root, patch_size = 14, pose_num = 3, margin_x = 0.3, margin_up = 0.05, margin_down = 0.15):
    try:
        pts = mio.import_landmark_file(gt_dir + name[:-3] + 'pts')
    except:
        return 'no gt'

    # make sure that the bb exists
    try:
        bb = mio.import_landmark_file(bb_root + name[:-3] + 'pts')
    except:
        return 'no bb'

    # verify reasonable size
    if not (bb.lms.points[2][0] - bb.lms.points[0][0] > 10 and
                        bb.lms.points[2][1] - bb.lms.points[0][1] > 10):
        return 'bb too small'

    # verify that the patches are within the bb + margins:
    # [bb left] - margin_ratio*width < [gt left] - patch_ext
    # and similar to all directions
    ### prev - marg = margin_x - patch_size / BB_SIZE / 2

    # patch_ext is the patch length beyond the actual landmark. The whole patch around the gt flms should be included in the bb+margins
    patch_ext = (patch_size/2) / common_params.BB_SIZE
    bb_w = bb.lms.range()[1]
    bb_h = bb.lms.range()[0]
    if not(bb.lms.points[0][0] - margin_up*bb_h < pts.lms.bounding_box().points[0][0] - patch_ext*bb_h and
            bb.lms.points[0][1]  - margin_x*bb_w < pts.lms.bounding_box().points[0][1] - patch_ext*bb_w and
            bb.lms.points[2][0] + margin_down*bb_h > pts.lms.bounding_box().points[2][0] + patch_ext*bb_h and
            bb.lms.points[2][1] + margin_x*bb_w > pts.lms.bounding_box().points[2][1] + patch_ext*bb_w):
        return 'patches not included'

    try:
        with open(pose_root + name[:-3] + 'txt') as f:
            pose = float(next(f))
            if pose < -60 or pose >= 60:
                return 'extreme yaw'
    except:
        return 'no yaw'

    return ''
########################################################################################################################

def build_image_pyr(image, init_shape, PYR_LEVELS):

    pyr = image.gaussian_pyramid(PYR_LEVELS)
    all_levels = [im for im in pyr]
    image_pyramid = [im.pixels.transpose(1, 2, 0).astype('float32') for im in all_levels]
    initial = image.landmarks['__initial'].lms
    # shape_pyramid = [im.landmarks['PTS'].lms.points.astype('float32') for im in all_levels]
    # init_shape.points = (init_shape.points / INITIAL_POINTS_common_params.BB_SIZE ) * BB_SIZE
    init_shape_pyramid = [initial.points / (2**d) for d in range(PYR_LEVELS)]

    return image_pyramid, init_shape_pyramid

########################################################################################################################
def load_image_for_eval_pyr(name, data_dir, gt_dir, bb_root, yaw_root, pitch_root, roll_root, reference_shapes, yaw_range,
                        pitch_range, margin=0.3, roll_support=False, grayscale=False):
    #print('in load_image_for_eval, grayscale='+str(grayscale))
    im = mio.import_image(data_dir + name)
    # gil - why gt in eval?
    # im.landmarks['PTS'] = mio.import_landmark_file(gt_dir + name[:-3] + 'pts')
    # group = im.landmarks.group_labels[0]
    im.landmarks['bb'] = mio.import_landmark_file(bb_root + name[:-3] + 'pts')

    # get reference shape
    # gil - remove
    pose_ind = 0
    if not common_params.SINGLE_POSE_ONLY:
        if np.any(yaw_range):
            with open(yaw_root + name[:-3] + 'txt') as f1:
                yaw = float(next(f1))
            if np.any(pitch_range):
                with open(pitch_root + name[:-3] + 'txt') as f1:
                    pitch = float(next(f1))
                pose_ind =  get_yaw_pitch_ind(yaw, pitch, yaw_range, pitch_range)
            else:
                pose_ind = get_pose_ind(yaw, yaw_range)
        else: # HFD
            with open(yaw_root + name[:-3] + 'txt') as f:
                pose_ind = int(next(f))

    # gil

    if pose_ind == 3:
        pose_ind = 0

    # gil - experiment with frontal pose init only
    # pose_ind = 0
    if common_params.SINGLE_POSE_ONLY:
        pose_ind = 0

    reference_shape = PointCloud(reference_shapes[pose_ind])

    roll = 0
    if (roll_support):
        with open(roll_root + name[:-3] + 'txt') as f:
            roll = float(next(f))

    # NOTE: symetric margins, simpler rotation around the center
    im, Tmat, min_x, min_y = preprocess_image_menpo(im.copy(), reference_shape.copy(), roll, roll_support,
                                                  target = common_params.BB_SIZE, margin = margin, grayscale = grayscale)

    im = menpo.feature.normalize_std(im)

    # gil - normalize the image
    #im = menpo.feature.normalize_std(im)
    #print('in load_image_for_eval, im.shape = '+str(im.shape))

    PYR_LEVELS = 3
    # gil - the init shape is reduced to the top level pyramid dimension
    image_pyramid, init_shape_pyramid = build_image_pyr(im, reference_shape, PYR_LEVELS)

    return image_pyramid[0].astype(np.float32).copy(), \
           image_pyramid[1].astype(np.float32).copy(), \
           image_pyramid[2].astype(np.float32).copy(),\
           init_shape_pyramid[0].astype(np.float32).copy(), \
           init_shape_pyramid[1].astype(np.float32).copy(), \
           init_shape_pyramid[2].astype(np.float32).copy(), \
           np.array(Tmat).astype(np.float32)
    # try:
    #     lms = im.landmarks[group].lms
    # except:
    #     lms = im.landmarks['__initial'].lms
    #
    # initial = im.landmarks['__initial'].lms
    #
    # # if the image is greyscale then convert to rgb.
    # if not(grayscale):
    #     pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)
    # else:
    #     pixels = im.pixels.transpose(1, 2, 0)
    # #    print(pixels.shape)
    #
    # gt_truth = lms.points.astype(np.float32)
    # estimate = initial.points.astype(np.float32)
    #
    # return pixels.astype(np.float32).copy(), gt_truth, estimate, np.array(Tmat).astype(np.float32)


    PYR_LEVELS = 4
    # gil - the init shape is reduced to the top level pyramid dimension
    image_pyramid, init_shape_pyramid = build_image_pyr(im, reference_shape, PYR_LEVELS)
    if not (grayscale):
        pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)
    else:
        pixels = im.pixels.transpose(1, 2, 0)
    #    print(pixels.shape)

    return pixels.astype(np.float32).copy()
    return image_pyramid[0], image_pyramid[1], image_pyramid[2], image_pyramid[3], init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], \
           init_shape_pyramid[3], Tmat

    # print('in load_image_for_eval, im.shape = '+str(im.shape))

    try:
        lms = im.landmarks[group].lms
    except:
        lms = im.landmarks['__initial'].lms

    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    if not (grayscale):
        pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)
    else:
        pixels = im.pixels.transpose(1, 2, 0)
    #    print(pixels.shape)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)

    return pixels.astype(np.float32).copy(), gt_truth, estimate, np.array(Tmat).astype(np.float32)


########################################################################################################################

def load_image_for_eval(name, data_dir, gt_dir, bb_root, yaw_root, pitch_root, roll_root, reference_shapes, yaw_range,
                        pitch_range, margin = 0.3, roll_support = False, grayscale = False):

    #print('in load_image_for_eval, grayscale='+str(grayscale))
    im = mio.import_image(data_dir + name)
    # gil - why gt in eval?
    # im.landmarks['PTS'] = mio.import_landmark_file(gt_dir + name[:-3] + 'pts')
    # group = im.landmarks.group_labels[0]
    im.landmarks['bb'] = mio.import_landmark_file(bb_root + name[:-3] + 'pts')

    # get reference shape
    if np.any(yaw_range):
        with open(yaw_root + name[:-3] + 'txt') as f1:
            yaw = float(next(f1))
        if np.any(pitch_range):
            with open(pitch_root + name[:-3] + 'txt') as f1:
                pitch = float(next(f1))
            pose_ind =  get_yaw_pitch_ind(yaw, pitch, yaw_range, pitch_range)
        else:
            pose_ind = get_pose_ind(yaw, yaw_range)
    else: # HFD
        with open(yaw_root + name[:-3] + 'txt') as f:
            pose_ind = int(next(f))

    # gil
    if pose_ind == 3:
        pose_ind = 0

    # gil - experiment with frontal pose init only
    pose_ind = 0
    reference_shape = PointCloud(reference_shapes[pose_ind])

    roll = 0
    if (roll_support):
        with open(roll_root + name[:-3] + 'txt') as f:
            roll = float(next(f))

    # NOTE: symetric margins, simpler rotation around the center
    im, Tmat, min_x, min_y = preprocess_image_menpo(im.copy(), reference_shape.copy(), roll, roll_support,
                                                  target = common_params.BB_SIZE, margin = margin, grayscale = grayscale)



    #print('in load_image_for_eval, im.shape = '+str(im.shape))

    try:
        lms = im.landmarks[group].lms
    except:
        lms = im.landmarks['__initial'].lms

    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    if not(grayscale):
        pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)
    else:
        pixels = im.pixels.transpose(1, 2, 0)
    #    print(pixels.shape)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)

    return pixels.astype(np.float32).copy(), gt_truth, estimate, np.array(Tmat).astype(np.float32)

def calc_bb_center(bb, margin_ratio_down, margin_ratio_up):
    h, w = bb[3] - bb[1], bb[2] - bb[0]
    h_bb = h / (1 + margin_ratio_down + margin_ratio_up)
    h_center = bb[1] + h_bb * (0.5 + margin_ratio_up)
    w_center = bb[0] + w * 0.5  # w margin is symetric

    return (h_center, w_center)

def distort_color(image, thread_id=0, stddev=0.1, scope=None, grayscale=False):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            if not grayscale:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)

        elif color_ordering == 1:
            # gil - original values
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            #image = tf.image.random_brightness(image, max_delta=16. / 255.)
            #image = tf.image.random_contrast(image, lower=0.75, upper=1.25)

            if not grayscale:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)

        image += tf.random_normal(
            tf.shape(image),
            stddev=stddev,
            dtype=tf.float64,
            seed=42,
            name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

########################################################################################################################

def get_pyramid_shapes(name):
    _padded_im, _shape, _init, _bb = load_image_for_train_hfd(name, FLAGS.data_dir, FLAGS.gt_path,
                                                                            FLAGS.pose_root, FLAGS.bb_root,
                                                                            reference_shape)
    im = menpo.image.Image(_padded_im.transpose(2, 0, 1), copy=False)
    lms = _shape
    init_shape = _init
    im.landmarks['PTS'] = _shape

    shape = im.landmarks['PTS'].lms.points.astype('float32')
    init_shape = init_shape.points.astype('float32')
    # gil - we construct 4 pyramid levels and

    PYR_LEVELS = 4
    # gil - the init shape is reduced to the top level pyramid dimension
    image_pyramid, shape_pyramid, init_shape_pyramid = build_image_pyr(im, init_shape, PYR_LEVELS)

    out = [v.shape for v in image_pyramid]
    return out

########################################################################################################################

def batch_inputs_pyr(files,data_dir, gt_dir, bb_root, yaw_root, pitch_root, roll_root,
                 reference_shape, yaw_range = [], pitch_range = [],
                 margin = 0.3,
                 batch_size=1, is_training=False, num_landmarks=165,
                roll_support = True, grayscale = False):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
      num_landmarks: the number of landmarks in the training images.
      mirror_image: mirrors the image and landmarks horizontally.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """

   # files = tf.concat(0, [map(str, sorted(Path(d).parent.glob(Path(d).name)))
   #                         for d in paths])



    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=100)

    name = filename_queue.dequeue()
    print('in batch_inputs: '+name)

    PYR_LEVEL = 3
    image_pyramid = [None] * PYR_LEVEL
    init_shape_pyramid = [None] * PYR_LEVEL

    DEBUG = 0
    if DEBUG:
        name = '/IBUG/IBUG_image_024_1_0.jpg'
        image_pyramid[0], image_pyramid[1], image_pyramid[2],\
        init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], Tmat\
        = load_image_for_eval_pyr( name, data_dir, gt_dir, bb_root,yaw_root, pitch_root, roll_root,
                                           reference_shape, yaw_range, pitch_range, margin, roll_support, grayscale)
    else:
        # name = '0028.jpg'
        # image_pyramid[0], image_pyramid[1], image_pyramid[2], image_pyramid[3], init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], init_shape_pyramid[3], Tmat = tf.py_func(load_image_for_eval_pyr,[name, data_dir, gt_dir, bb_root,yaw_root, pitch_root, roll_root, reference_shape, yaw_range, pitch_range, margin, roll_support, grayscale], [tf.float32, tf.float32, tf.float32,tf.float32,tf.float32,tf.float32, tf.float32,tf.float32, tf.float32])
        image_pyramid[0], image_pyramid[1],image_pyramid[2], \
        init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], Tmat = tf.py_func(load_image_for_eval_pyr,
                                                [name, data_dir, gt_dir, bb_root, yaw_root, pitch_root, roll_root,
                                                 reference_shape, yaw_range, pitch_range, margin, roll_support,
                                                 grayscale],
                                                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

        # image = tf.py_func(load_image_for_eval_pyr,[name, data_dir, gt_dir,
        #                                                                                         bb_root, yaw_root,
        #                                                                                         pitch_root, roll_root,
        #                                                                                         reference_shape,
        #                                                                                         yaw_range, pitch_range,
        #                                                                                         margin, roll_support,
        #                                                                                         grayscale],[tf.float32])


    if grayscale:
       num_channels = 1
    else:
        num_channels = 3
   # print('num channels = '+str(num_channels))

    # 3 channels - rgb, 1 channel - y only
    # image.set_shape([None, None, num_channels])
    # gil
    Tmat.set_shape([3, 3])

#    bb_src.set_shape([4])
#    bb_dst.set_shape([4])
#    roll.set_shape([])

    # if is_training:
    #     image = distort_color(image)

    # lms = tf.reshape(lms, [num_landmarks, 2])
    # lms_init = tf.reshape(lms_init, [num_landmarks, 2])
    # image_shapes = get_pyramid_shapes('/raid/algo/SOCVISION_SLOW/FLM/flm_db/images/test_set/1.jpg')

    # image_pyramid[0].set_shape([408, 408, 3])
    #

    for index, im_level in enumerate(image_pyramid):
        im_level.set_shape([None, None, 3])
    for init_shape_level in init_shape_pyramid:
        init_shape_level.set_shape([num_landmarks, 2])
    #
    image_pyramids = [None]*PYR_LEVEL
    init_shape_pyramids = [None]*PYR_LEVEL

    filenames, image_pyramids[0], image_pyramids[1], image_pyramids[2], \
    init_shape_pyramids[0], init_shape_pyramids[1], init_shape_pyramids[2], Tmats \
     = tf.train.batch(
        [name, image_pyramid[0], image_pyramid[1], image_pyramid[2], init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], Tmat],
        # FLAGS.batch_size,
        len(files),
        num_threads=1,
        capacity=1000,
        enqueue_many=False,
        dynamic_pad=True)

    # with tf.Session() as sess:
    #     saver.restore(sess, FLAGS.checkpoint_file)
    #
    #     coord = tf.train.Coordinator()
    #
    #     threads = []
    #     for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                          start=True))
    #
    #     num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    #     # Counts the number of correct predictions.
    #     errors = []
    #     predictions = []
    #     fNames = []
    #     gt = []
    #
    #     total_sample_count = num_iter * FLAGS.batch_size
    #     step = 0
    #
    #     print('%s: starting evaluation' % datetime.now())
    #     start_time = time.time()
    #     if step < num_iter and not coord.should_stop():
    #         filename, GTlms, pred, rmse, pred0 = sess.run([filenames, avg_pred, rmse_op, preds['0']])
    #

    # filenames,image_pyramids[0], image_pyramids[1], image_pyramids[2], image_pyramids[3], \
    # init_shape_pyramids[0], init_shape_pyramids[1], init_shape_pyramids[2], init_shape_pyramids[3], Tmats = tf.train.batch(
    #     [name,image_pyramid[0], image_pyramid[1], image_pyramid[2], image_pyramid[3], init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2],
    #      init_shape_pyramid[3], Tmat],
    #     FLAGS.batch_size,
    #     num_threads=1,
    #     capacity=1000,
    #     enqueue_many=False,
    #     dynamic_pad=False)

    return filenames, image_pyramids, init_shape_pyramids, Tmats
    images, lms, inits, shapes, filenames, Tmats = tf.train.batch(
        [image, lms, lms_init, tf.shape(image), name, Tmat],
        batch_size=batch_size,
        num_threads=4 if is_training else 1,
        capacity=1000,
        enqueue_many=False,
        dynamic_pad=True)

    return images, lms, inits, shapes, filenames, Tmats



########################################################################################################################

def batch_inputs(files,data_dir, gt_dir, bb_root, yaw_root, pitch_root, roll_root,
                 reference_shape, yaw_range = [], pitch_range = [],
                 margin = 0.3,
                 batch_size=1, is_training=False, num_landmarks=165,
                roll_support = True, grayscale = False):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
      num_landmarks: the number of landmarks in the training images.
      mirror_image: mirrors the image and landmarks horizontally.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """

   # files = tf.concat(0, [map(str, sorted(Path(d).parent.glob(Path(d).name)))
   #                         for d in paths])

    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=100)

    name = filename_queue.dequeue()
    print('in batch_inputs: '+name)
    image, lms, lms_init, Tmat = tf.py_func(load_image_for_eval,
                                          [name, data_dir, gt_dir, bb_root,yaw_root, pitch_root, roll_root,
                                           reference_shape, yaw_range, pitch_range, margin, roll_support, grayscale],
                                          [tf.float32, tf.float32, tf.float32, tf.float32])

    if grayscale:
       num_channels = 1
    else:
        num_channels = 3
   # print('num channels = '+str(num_channels))

    # 3 channels - rgb, 1 channel - y only
    image.set_shape([None, None, num_channels])
    Tmat.set_shape([3, 3])

#    bb_src.set_shape([4])
#    bb_dst.set_shape([4])
#    roll.set_shape([])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, [num_landmarks, 2])
    lms_init = tf.reshape(lms_init, [num_landmarks, 2])


    images, lms, inits, shapes, filenames, Tmats = tf.train.batch(
        [image, lms, lms_init, tf.shape(image), name, Tmat],
        batch_size=batch_size,
        num_threads=4 if is_training else 1,
        capacity=1000,
        enqueue_many=False,
        dynamic_pad=True)

    return images, lms, inits, shapes, filenames, Tmats



if __name__ == '__main__':
    filenames_file = '/raid/algo/SOCVISION_SLOW/FLM/flm_db/db_views/NFD/train_sets/all_nfd_filenames.txt'
    gt_dir = '/raid/algo/SOCVISION_SLOW/FLM/flm_db/gt/165_silhouette/300W_LP/'
    bb_dir = '/raid/algo/SOCVISION_SLOW/FLM/300W_LP/semi_frontal/300W_LP_semi_frontal_NFD_v2_new/bbs/'
    pose_dir = ''

    with open(filenames_file) as f:
        files = f.read().split()

    build_reference_shape_per_pose_by_files(files, gt_dir, bb_dir, pose_dir, num_of_poses=1)