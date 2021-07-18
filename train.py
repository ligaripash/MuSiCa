# New train - validation split

from datetime import datetime
import data_provider
import mdm_model
import numpy as np
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import menpo.io as mio
from menpo.shape.pointcloud import PointCloud
import math
import shutil
import glob

from train_params import *
import common_params
import cv2

def load_reference_shape(filenames):

   if FLAGS.with_pose == 0:
        reference_shape_file = FLAGS.reference_shape_path + 'reference_shape.pkl'
        reference_file_exists = os.path.isfile(reference_shape_file)
   else:
       reference_file_format = FLAGS.reference_shape_path + 'reference_shape_%d.pkl'
       reference_file_exists = os.path.isfile(reference_file_format % 0) and os.path.isfile(reference_file_format % 1) and os.path.isfile(reference_file_format % 2)

   if reference_file_exists:
        try:
            reference_shape = data_provider.load_reference_shape(FLAGS.reference_shape_path, FLAGS.with_pose)
            # reference_subset = reference_shape.points[common_params.PATCH_INDEXES]
            # reference_subset = PointCloud(reference_subset)
            return reference_shape
        except:
            print('bad reference shape - building new')

   if FLAGS.with_pose == 0:
       # todo (noga)
   #      reference_shape = PointCloud(data_provider.build_reference_shape_v2(filenames, bb_root_main=FLAGS.bb_root))
   # else:
        reference_shape = data_provider.build_reference_shape_per_pose_by_files(filenames, FLAGS.gt_path, FLAGS.bb_root, FLAGS.pose_root, num_of_poses=common_params.INIT_POSE_COUNT)

   data_provider.save_reference_shape(reference_shape, FLAGS.with_pose, FLAGS.reference_shape_path)

   return reference_shape


########################################################################################################################


def build_image_pyr(image, init_shape, PYR_LEVELS):

    pyr = image.gaussian_pyramid(PYR_LEVELS)
    all_levels = [im for im in pyr]
    image_pyramid = [im.pixels.transpose(1, 2, 0).astype('float32') for im in all_levels]
    shape_pyramid = [im.landmarks['PTS'].lms.points.astype('float32') for im in all_levels]
    init_shape_pyramid = [init_shape / (2**d) for d in range(PYR_LEVELS)]

    return image_pyramid, shape_pyramid, init_shape_pyramid

########################################################################################################################

def calc_augmentation_parameters(width):
    scale_range = (np.random.randint(100 - SCALE_AUG_FACTOR,100 + SCALE_AUG_FACTOR )) / 100.0
    translation_range = (np.random.randint(0, TRANSLATION_AUG_FACTOR)) / 100.0
    translation = translation_range * width

    return scale_range, translation_range


########################################################################################################################

def get_latest_checkpoint(pretrainedPath):

    def func(x):
        base_name = os.path.basename(x)
        d = int(base_name.split('-')[1][:-5])
        return d

    all_models = glob.glob(pretrainedPath + '/*model.ckpt*meta')
    if len(all_models) == 0:
        return '',0
    sorted_files = sorted(all_models, key=func)
    last_model = sorted_files[-1]
    step = int(func(last_model))
    last_model = last_model[:-5]
    return last_model, step

########################################################################################################################

def train(scope=''):
    idx = 0
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device(FLAGS.train_device):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # Calculate the learning rate schedule.
        decay_steps = 15000
        ADAM = False
        if FLAGS.optimizer == "ADAM":
        # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                            global_step,
                                            decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True)

        # Create an optimizer that performs gradient descent.
            opt = tf.train.AdamOptimizer(lr)
        elif FLAGS.optimizer == "SGM":
            starter_lr = 0.1
            lr = tf.train.exponential_decay(starter_lr,
                                            global_step,
                                            30000,
                                            0.96,
                                            staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)

        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = FLAGS.num_preprocess_threads

        # create filenames queue
        with open(FLAGS.train_filenames, 'rt') as f:
            filenames = [name.strip() for name in f.readlines()]
        filenames_queue = tf.train.string_input_producer(filenames, num_epochs = FLAGS.num_epochs, shuffle = True)

        reference_shape = load_reference_shape(filenames)

        def get_pyramid_shapes(name):
            # _padded_im, _shape, _init, _bb = data_provider.load_image_for_train_hfd(name, FLAGS.data_dir, FLAGS.gt_path,
            #                                                                         FLAGS.pose_root, FLAGS.bb_root,
            #                                                                         reference_shape)

            # _padded_im, _shape, _init, _bb = data_provider.load_image_for_train_hfd(name, FLAGS.data_dir, FLAGS.gt_path,
            #                                                                         FLAGS.pose_root, FLAGS.bb_root,
            #                                                                         reference_shape)

            _padded_im, _shape, _init, _bb =  data_provider.load_image_for_train_new(name, FLAGS.image_dir,
                                                                                     FLAGS.gt_path, FLAGS.bb_root,
                                                                                     reference_shape, FLAGS.margin, roll_support = False)

            _init_sub_sample = PointCloud([_init.points[index] for index in common_params.FLM_INDEXES])
            _shape_sub_sample = PointCloud([_shape.points[index] for index in common_params.FLM_INDEXES])
            im = menpo.image.Image(_padded_im.transpose(2, 0, 1), copy=False)
            lms = _shape_sub_sample
            init_shape = _init_sub_sample
            im.landmarks['PTS'] = _shape

            shape = im.landmarks['PTS'].lms.points.astype('float32')
            init_shape = init_shape.points.astype('float32')
            # gil - we construct 4 pyramid levels and

            # PYR_LEVELS = 4
            # gil - the init shape is reduced to the top level pyramid dimension
            image_pyramid, shape_pyramid, init_shape_pyramid = build_image_pyr(im, init_shape, common_params.PYRAMID_HEIGHT)

            out = [v.shape for v in image_pyramid]
            return out

        def read_image_pyr(name, rotation_stddev=10, y_only=False, add_noise=False):

            # load image & metadata
            #print (name)
            _padded_im, _shape, _init, _bb =  data_provider.load_image_for_train_new(name, FLAGS.image_dir,
                                                                                     FLAGS.gt_path, FLAGS.bb_root,
                                                                                     reference_shape, FLAGS.margin, roll_support = False)
            # augment
            # gil - y channel only
            if add_noise:
                _padded_im = data_provider.distort_color(_padded_im)
            if y_only:
                gray = cv2.cvtColor(_padded_im, cv2.COLOR_RGB2GRAY)
                _padded_im = gray
            else:
                im = menpo.image.Image(_padded_im.transpose(2, 0, 1), copy=False)
            # cv2.imwrite('/home/gilsh/tmp/padded_im.jpg', _padded_im * 255)
            # im = _padded_im

            # im = menpo.image.Image(_padded_im, copy=False)
            _init_sub_sample = PointCloud([_init.points[index] for index in common_params.FLM_INDEXES])
            _shape_sub_sample = PointCloud([_shape.points[index] for index in common_params.FLM_INDEXES])

            lms = _shape_sub_sample
            init_shape = _init_sub_sample
            im.landmarks['PTS'] = _shape_sub_sample


            #in wflw we skip mirror
            # if np.random.rand() < .5:
            #     im = utils.mirror_image(im)
            # gil - rotation augmentation needs fd rect update. disable for now
            if np.random.rand() < 0.0:
                #theta = np.random.normal(scale=rotation_stddev)
                max_rotation = 30
                theta = np.random.uniform(-max_rotation, max_rotation)
                rot = menpo.transform.rotate_ccw_about_centre(lms, theta)
                im = im.warp_to_shape(im.shape, rot)

            # if np.random.rand() < .1:
            #     init_shape = lms
            # Augmentation is done in data_provider
            if np.random.rand() < 0.0:
                c_x = init_shape.points[:, 0] - np.mean(init_shape.points[:, 0])
                c_y = init_shape.points[:, 1] - np.mean(init_shape.points[:, 1])
                # f_x = (np.max(c_x) - np.min(c_x) + s_x) / (np.max(c_x) - np.min(c_x))
                # f_y = (np.max(c_y) - np.min(c_y) + s_y) / (np.max(c_y) - np.min(c_y))
                width = np.max(c_x) - np.min(c_x)
                scale, trans = calc_augmentation_parameters(width)
                init_shape.points[:, 0] = c_x * scale + np.mean(init_shape.points[:, 0]) + trans
                init_shape.points[:, 1] = c_y * scale + np.mean(init_shape.points[:, 1]) + trans

            image = im.pixels.transpose(1, 2, 0).astype('float32')
            shape = im.landmarks['PTS'].lms.points.astype('float32')
            init_shape = init_shape.points.astype('float32')
            # gil - we construct 4 pyramid levels and

            # PYR_LEVELS = 3
            # gil - the init shape is reduced to the top level pyramid dimension
            image_pyramid, shape_pyramid, init_shape_pyramid = build_image_pyr(im, init_shape, common_params.PYRAMID_HEIGHT)

            return image_pyramid[0], image_pyramid[1], image_pyramid[2], shape_pyramid[0], \
                   shape_pyramid[1], shape_pyramid[2], init_shape_pyramid[0], init_shape_pyramid[1], \
                   init_shape_pyramid[2]
            # return image_pyramid[0], image_pyramid[1],image_pyramid[2], image_pyramid[3], shape_pyramid[0], shape_pyramid[1], shape_pyramid[2], shape_pyramid[3],init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], init_shape_pyramid[3]
            # return image_pyramid[0], image_pyramid[1], image_pyramid[2], image_pyramid[3],\
            #        shape_pyramid[0], shape_pyramid[1], shape_pyramid[2], shape_pyramid[3],\
            #        init_shape_pyramid[0],  init_shape_pyramid[1], init_shape_pyramid[2], init_shape_pyramid[3]



        ########################################################################################################################


        def read_image(name, rotation_stddev=10, y_only=False, add_noise=False):
            # load image & metadata
            _padded_im, _shape, _init, _bb = data_provider.load_image_for_train_hfd(name, FLAGS.data_dir, FLAGS.gt_path, FLAGS.pose_root, FLAGS.bb_root,
                                                                                reference_shape)
            # augment
            # gil - y channel only
            if add_noise:
                _padded_im = data_provider.distort_color(_padded_im)

            if y_only:
                gray = cv2.cvtColor(_padded_im, cv2.COLOR_RGB2GRAY)
                _padded_im = gray
            else:
                im = menpo.image.Image(_padded_im.transpose(2, 0, 1), copy=False)
            # cv2.imwrite('/home/gilsh/tmp/padded_im.jpg', _padded_im * 255)
            # im = _padded_im
            # im = menpo.image.Image(_padded_im.transpose(2, 0, 1), copy=False)
            im = menpo.image.Image(_padded_im, copy=False)
            lms = _shape
            init_shape = _init
            im.landmarks['PTS'] = _shape

            if np.random.rand() < -.2:
               im = utils.mirror_image(im)

            if np.random.rand() < .5:
              theta = np.random.normal(scale=rotation_stddev)
              rot = menpo.transform.rotate_ccw_about_centre(lms, theta)
              im = im.warp_to_shape(im.shape, rot)

            if np.random.rand() < .1:
                init_shape = lms

            if np.random.rand() < .2:
                t_x = np.round(AUG_FACTOR * np.random.rand() - AUG_FACTOR/2)
                t_y = np.round(AUG_FACTOR * np.random.rand() - AUG_FACTOR/2)
                s_x = np.round(AUG_FACTOR * np.random.rand() - AUG_FACTOR/2)
                s_y = np.round(AUG_FACTOR * np.random.rand() - AUG_FACTOR/2)
                c_x = init_shape.points[:, 0] - np.mean(init_shape.points[:, 0])
                c_y = init_shape.points[:, 1] - np.mean(init_shape.points[:, 1])
                f_x = (np.max(c_x) - np.min(c_x) + s_x) / (np.max(c_x) - np.min(c_x))
                f_y = (np.max(c_y) - np.min(c_y) + s_y) / (np.max(c_y) - np.min(c_y))
                init_shape.points[:, 0] = c_x * f_x + np.mean(init_shape.points[:, 0]) + t_x
                init_shape.points[:, 1] = c_y * f_y + np.mean(init_shape.points[:, 1]) + t_y

            image = im.pixels.transpose(1, 2, 0).astype('float32')
            shape = im.landmarks['PTS'].lms.points.astype('float32')
            init_shape = init_shape.points.astype('float32')

            # image, shape = gt lms, init_shape - reference_shape that matches the pose
            # gil
            # print image.shape
            # cv2.imwrite('/home/gilsh/tmp/padded_im.jpg', image * 255)
            return image, shape, init_shape

        # get image
        name = filenames_queue.dequeue()
        # gil - image pyramid
        # image, shape, init_shape = tf.py_func(read_image, [name], [tf.float32, tf.float32, tf.float32])
        #image = data_provider.distort_color(image)

        DEBUG = 0
        if DEBUG:
            #name = 'LFPW_trainset/image_0409.png'
            name = '44--Aerobics/44_Aerobics_Aerobics_44_973_0.jpg'
            while (1):
                image_shapes = get_pyramid_shapes(name)
            image_pyramid, init_shape = read_image_pyr(name)

        # name = '/AFLW/image00002.jpg'
        # image_shapes = get_pyramid_shapes('DB100kSmileBlinkFaces/chunk00002/0000000001001_02.jpg')


        image_pyramid = [None]*common_params.PYRAMID_HEIGHT
        shape_pyramid = [None]*common_params.PYRAMID_HEIGHT
        init_shape_pyramid = [None]*common_params.PYRAMID_HEIGHT

        # image_pyramid[0], image_pyramid[1], image_pyramid[2], image_pyramid[3], shape_pyramid[0], shape_pyramid[1], shape_pyramid[2], shape_pyramid[3],init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], init_shape_pyramid[3] = tf.py_func(read_image_pyr, [name],[tf.float32, tf.float32, tf.float32, tf.float32,tf.float32, tf.float32, tf.float32, tf.float32,tf.float32, tf.float32, tf.float32, tf.float32])
        image_pyramid[0], image_pyramid[1], image_pyramid[2], shape_pyramid[0], shape_pyramid[1], \
        shape_pyramid[2], init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2], \
        = tf.py_func(read_image_pyr, [name],
                                           [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                            tf.float32, tf.float32, tf.float32])
        lms_shape = [common_params.FLM_COUNT, 2]
        for index, im_level in enumerate(image_pyramid):
            # im_level.set_shape(image_shapes[index])
            im_level.set_shape((None, None, 3))
            # im_level.set_shape(data_provider.get_hfd_image_shape(channel_count=3, pyr_level=index))
        for shape_level in shape_pyramid:
            shape_level.set_shape(lms_shape)


        for init_shape_level in init_shape_pyramid:
            init_shape_level.set_shape(lms_shape)


        image_pyramids = [None]*common_params.PYRAMID_HEIGHT
        shape_pyramids = [None]*common_params.PYRAMID_HEIGHT
        init_shape_pyramids = [None]*common_params.PYRAMID_HEIGHT
        image_pyramids[0], image_pyramids[1], image_pyramids[2],  \
        shape_pyramids[0], shape_pyramids[1], shape_pyramids[2],  \
        init_shape_pyramids[0], init_shape_pyramids[1], init_shape_pyramids[2] = tf.train.batch([image_pyramid[0], image_pyramid[1], image_pyramid[2], shape_pyramid[0], shape_pyramid[1], shape_pyramid[2], init_shape_pyramid[0], init_shape_pyramid[1], init_shape_pyramid[2]],
                                                                             FLAGS.batch_size,
                                                                             num_threads=num_preprocess_threads,
                                                                             capacity=5000,
                                                                             enqueue_many=False,
                                                                             dynamic_pad=True,
                                                                             name='batch')

        # image_pyramids, shape_pyramids, init_shape_pyramids = tf.train.batch([image_pyramid, shape_pyramid, init_shape_pyramid],
        #                                     FLAGS.batch_size,
        #                                     num_threads=num_preprocess_threads,
        #                                     capacity=5000,
        #                                     enqueue_many=False,
        #                                     dynamic_pad=False,
        #                                     name='batch')


        print('Defining model...')
        with tf.device(FLAGS.train_device):
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            predictions, dxs, endpoints, flm_predictions, hidden_states = \
                mdm_model.model_conv3_pyramids(image_pyramids,
                                               init_shape_pyramids,
                                               common_params.PATCH_INDEXES,
                                               num_iterations=common_params.ITERATIONS_COUNT,
                                               num_flms=common_params.FLM_COUNT,
                                               patch_shape=(FLAGS.patch_size, FLAGS.patch_size), num_channels=3,
                                               dropout=False)

            # predictions, dxs, _, flm_predictions, hidden_states = mdm_model.model_conv3(images, inits, num_iterations=4, num_patches=FLAGS.num_of_flms,
            #                                                                             patch_shape=(FLAGS.patch_size, FLAGS.patch_size), num_channels=3)

            total_loss = 0
            loss_per_iteration_for_first_face_in_batch = np.array([])
            loss_per_iteration = np.array([])
            estimated_error_per_iteration = np.array([])
            for i, dx in enumerate(dxs):
                # dx is given relative to level 0
                if not (FLAGS.weighted_loss):
                    if FLAGS.loss == 'L1':
                        norm_error = mdm_model.L1_loss(dx + init_shape_pyramids[0], shape_pyramids[0],
                                                               common_params.FLM_COUNT)
                    else:
                        norm_error = mdm_model.normalized_rmse(dx + init_shape_pyramids[0], shape_pyramids[0], common_params.FLM_COUNT)
                    # norm_error = mdm_model.normalized_mse(dx + init_shape_pyramids[0], shape_pyramids[0],
                    #                                        FLAGS.num_of_flms)
                    predicted_error = endpoints['error_estimation_' + str(i) ]
                    predicted_error_error = tf.abs(norm_error - predicted_error)
                    # norm_error = norm_error + tf.abs(norm_error - predicted_error)
                else:
                    norm_error = mdm_model.normalized_rmse_weighted(dx + init_shape_pyramids[0], shape_pyramids[0], FLAGS.num_of_flms)

                tf.histogram_summary('errors', norm_error)
                loss = tf.reduce_mean(norm_error)
                predicted_error_loss = tf.reduce_mean(predicted_error_error)
                avg_predicted_error = tf.reduce_mean(predicted_error)

                loss_per_iteration_for_first_face_in_batch = np.append(loss_per_iteration_for_first_face_in_batch, norm_error[0])
                loss_per_iteration = np.append(loss_per_iteration, loss)
                estimated_error_per_iteration = np.append(estimated_error_per_iteration, avg_predicted_error)
                predicted_loss_weight = 0.05
                #gils - experiment without error estimation
                # predicted_loss_weight = 0.0
                total_loss += (1 - predicted_loss_weight) * loss + predicted_loss_weight * predicted_error_loss
                # total_loss += loss
                summaries.append(tf.scalar_summary('losses/step_{}'.format(i), loss))

            # Calculate the gradients for the batch of data
            grads = opt.compute_gradients(total_loss)
        #     gil - clip gradients
        #     grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]

        summaries.append(tf.scalar_summary('losses/total', total_loss))

        DEBUG = 0
        if DEBUG:
            image = np.zeros((1,64,64,3))
            p0 = np.zeros((1, 165,2))
            p1 = np.zeros((1, 165,2))
            p2 = np.zeros((1, 165, 2))
            out = utils.render_points_on_image(image, p0, p1, p2, 0.02)

        debug_pyr = [None] * common_params.ITERATIONS_COUNT
        for iter in range(common_params.ITERATIONS_COUNT):
            pyr_level = common_params.ITERATIONS_COUNT - iter - 1
            init = 'init_' + str(iter)
            final = 'final_' + str(iter)
            debug_pyr[iter] = tf.py_func(utils.render_points_on_image, [image_pyramids[pyr_level], endpoints[init], endpoints[final], shape_pyramids[pyr_level] ,loss_per_iteration_for_first_face_in_batch[iter]],[tf.float32])
            debug_pyr[iter][0].set_shape((1,None, None, 3))

            summary = tf.image_summary('level_' + str(iter), debug_pyr[iter][0])
            summaries.append(summary)
            # summary1 = tf.image_summary('level_1', debug_l1[0])
            # summary2 = tf.image_summary('level_2', debug_l2[0])

        # debug_l0 = tf.py_func(utils.render_points_on_image, [image_pyramids[2], endpoints['init_0'], endpoints['final_0'], shape_pyramids[2] ,loss_per_iteration_for_first_face_in_batch[0]],[tf.float32])
        # debug_l0[0].set_shape((1,None, None, 3))
        # debug_l1 = tf.py_func(utils.render_points_on_image, [image_pyramids[1], endpoints['init_1'], endpoints['final_1'], shape_pyramids[1],loss_per_iteration_for_first_face_in_batch[1]], [tf.float32])
        # debug_l1[0].set_shape((1,None, None, 3))
        # debug_l2 = tf.py_func(utils.render_points_on_image, [image_pyramids[0], endpoints['init_2'], endpoints['final_2'], shape_pyramids[0], loss_per_iteration_for_first_face_in_batch[2]], [tf.float32])
        # debug_l2[0].set_shape((1,None, None, 3))
        # # debug_l3 = tf.py_func(utils.render_points_on_image, [image_pyramids[0], endpoints['init_3'], endpoints['final_3'],shape_pyramids[0], loss_per_iteration_for_first_face_in_batch[3]], [tf.float32])
        # debug_l3[0].set_shape((1,None, None, 3))

        # pred_images, = tf.py_func(utils.batch_draw_landmarks,
        #                           [image_pyramids[0], predictions], [tf.float32])
        # gt_images, = tf.py_func(utils.batch_draw_landmarks, [image_pyramids[0], shape_pyramids[0]],
        #                         [tf.float32])

        # summary0 = tf.image_summary('level_0',debug_l0[0])
        # summary1 = tf.image_summary('level_1', debug_l1[0])
        # summary2 = tf.image_summary('level_2', debug_l2[0])
        # summary3 = tf.image_summary('level_3', debug_l3[0])


        summaries.append(tf.histogram_summary('dx', predictions - init_shape_pyramids[0]))

        # summaries.append(summary0)
        # summaries.append(summary1)
        # summaries.append(summary2)
        # summaries.append(summary3)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope)

        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', lr))


#         grad_check = tf.check_numerics(grads, 'Found nan in grad')
#         with tf.control_dependencies([grad_check]):
        # Add histograms for gradients.
        CLIP_GRADS = False
        if CLIP_GRADS:
            for grad, var in grads:
                if grad is not None:
                    # grad = tf.cond(tf.is_nan(grad), 0., grad)
                     summaries.append(tf.histogram_summary(var.op.name +
                                                          '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        # NOTE: Currently we are not using batchnorm in MDM.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = None)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_summary(summaries)
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')

        step = 0

        pretrainedPath = FLAGS.pretrained_model_checkpoint_path
        if pretrainedPath == '':
            pretrainedPath, step = get_latest_checkpoint(FLAGS.train_dir)
        # if FLAGS.pretrained_model_step != '':
        #     pretrainedPath = pretrainedPath + '-' + str(FLAGS.pretrained_model_step)
        #     # gil - if we fine tune we start training steps from scratch
        #     if not FLAGS.is_finetune:
        #         step = FLAGS.pretrained_model_step + 1 # pre-training from 99 to 101


        if pretrainedPath:
            assert tf.gfile.Exists(pretrainedPath)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, pretrainedPath)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), pretrainedPath))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

        # calc num steps
        train_size = len(filenames)
        batchs_in_epoch = int(train_size / FLAGS.batch_size)
        num_steps = batchs_in_epoch*FLAGS.num_epochs
        # start threads
        coord = tf.train.Coordinator()

        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))
        eval_file = FLAGS.train_dir + 'eval_loss.txt'
#        tf.train.start_queue_runners(filenames_queue)
        print('Starting training...')
        while step < num_steps and not coord.should_stop():
            # forward-backward step
            start_time = time.time()

            # gil
            # batch_images = sess.run([images])
            # for i in range(batch_images[0].shape[0]):
            #     t = np.array(batch_images[0][i]*255, dtype=np.uint8)
            #     t = np.reshape(t,(t.shape[0], t.shape[1]))
            #     cv2.imwrite('/mnt/home/tmp/im' + str(i) + '.png', t)
            if common_params.ITERATIONS_COUNT == 3:
                summary_str, _,loss_value, final_loss, \
                loss_it_0, loss_it_1, loss_it_2, \
                predicted_error_loss_0, predicted_error_loss_1 , predicted_error_loss_2 \
                    = sess.run([summary_op,
                                train_op,
                                loss,
                                total_loss,
                                loss_per_iteration[0],
                                loss_per_iteration[1],
                                loss_per_iteration[2],
                                estimated_error_per_iteration[0],
                                estimated_error_per_iteration[1],
                                estimated_error_per_iteration[2]])#, eval_loss])
            elif common_params.ITERATIONS_COUNT == 2:

                summary_str,\
                _,\
                loss_value,\
                final_loss, \
                loss_it_0,\
                loss_it_1, \
                predicted_error_loss_0, \
                predicted_error_loss_1 \
                    = sess.run([summary_op,
                                train_op,
                                loss,
                                total_loss,
                                loss_per_iteration[0],
                                loss_per_iteration[1],
                                estimated_error_per_iteration[0],
                                estimated_error_per_iteration[1]]) #, eval_loss])


                summary_str, _, loss_value, final_loss, loss_it_0, loss_it_1, = sess.run(
                    [summary_op, train_op, loss, total_loss, loss_per_iteration[0], loss_per_iteration[1]])  # , eval_loss])
            if WRITE_TB_DATA:
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # print loss

            examples_per_sec = FLAGS.batch_size / float(duration)
            if common_params.ITERATIONS_COUNT == 3:
                format_str = (
                    '%s: step %d, loss_final = %.2f, l_0 = %.2f, l_1 = %.2f, l_2 = %.2f, pred_error_0 = %.4f, pred_error_1 = %.2f, pred_error_3 = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, loss_it_0, loss_it_1, loss_it_2, predicted_error_loss_0,
                                    predicted_error_loss_1, predicted_error_loss_2,
                                    examples_per_sec, duration))

            elif common_params.ITERATIONS_COUNT == 2:
                format_str = (
                    '%s: step %d, loss_final = %.2f, l_0 = %.2f, l_1 = %.2f, pred_error_0 = %.4f, pred_error_1 = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, loss_it_0, loss_it_1, predicted_error_loss_0,
                                    predicted_error_loss_1, examples_per_sec, duration))

            # Save the model checkpoint periodically.
            if step % 300 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step = step + 1

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    train()
