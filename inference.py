"""A library to evaluate MDM on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import common_params
from inference_params import *
from datetime import datetime

import data_provider
import math
import menpo
import matplotlib
import mdm_model
import mdm_train
import numpy as np
import os.path
import tensorflow as tf
import time
import utils
import menpo.io as mio
import cv2
import mdm_eval_utils
import pickle
#import GCN

from menpo.io.output.base import export_landmark_file




def get_model_weights():
    weights = {}
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # tf.saved_model.loader.load(sess, 'meta' ,'/raid/algo/SOCVISION_SLOW/FLM/flm_db/experiments/sil/arcsoft_plus_300W_LP_ms_NFD' )
    for w in from_vars:
        transposed_weights = w.eval()
        if len(transposed_weights.shape) == 4:
            # biases are copied 'as is'
            transposed_weights = np.transpose(transposed_weights, (3, 2, 0, 1))
        elif len(transposed_weights.shape) == 2:
            # caffe weight matrix for inner product is transposed (check why..)
            transposed_weights = np.transpose(transposed_weights, (1, 0))
        weights[w.name] = transposed_weights

    return weights

########################################################################################################################

def dump_tf_activations(dump_dir):
    debug = False
    vals = {}

    vals['load_image'] = tf.get_default_graph().get_tensor_by_name("PyFunc:0").eval()
    orig_patches = tf.get_default_graph().get_tensor_by_name("ExtractPatches:0").eval()
    reshape_patches = np.reshape(orig_patches, (
    orig_patches.shape[1], orig_patches.shape[2], orig_patches.shape[3], orig_patches.shape[4]))
    vals['init_points'] = tf.get_default_graph().get_tensor_by_name("Reshape_1:0").eval()
    if (debug):
        # dump images:
        image = np.array(vals['load_image'] * 255, dtype=np.uint8)

        # dump image patches:
        for i in range(reshape_patches.shape[0]):
            patch = reshape_patches[i]
            patch_image = np.array(patch * 255, dtype=np.uint8)
            cv2.rectangle(image, (int(vals['init_points'][i][1]) - 7, int(vals['init_points'][i][0] - 7)),
                          (int(vals['init_points'][i][1] + 7), int(vals['init_points'][i][0] + 7)), 3)
            scm.imsave('tf_patch_' + str(i) + ".png", patch_image)

        scm.imsave('tf_load_image.png', image)

    # NHWC to NCHW
    patches_transposed = np.transpose(reshape_patches, (0, 3, 1, 2))
    vals['patches_data'] = patches_transposed

    conv_2d = tf.get_default_graph().get_tensor_by_name("Conv/BiasAdd:0").eval()
    conv_2d_transposed = np.transpose(conv_2d, (0, 3, 1, 2))
    vals['conv_1'] = conv_2d_transposed

    max_pool = tf.get_default_graph().get_tensor_by_name("MaxPool/MaxPool:0").eval()
    vals['pool_1'] = np.transpose(max_pool, (0, 3, 1, 2))

    conv_2_2d = tf.get_default_graph().get_tensor_by_name("Conv_1/BiasAdd:0").eval()
    vals['conv_2'] = np.transpose(conv_2_2d, (0, 3, 1, 2))

    max_pool_1 = tf.get_default_graph().get_tensor_by_name("MaxPool_1/MaxPool:0").eval()
    vals['pool_2'] = np.transpose(max_pool_1, (0, 3, 1, 2))

    crop = tf.get_default_graph().get_tensor_by_name("Slice:0").eval()
    vals['crop'] = np.transpose(crop, (0, 3, 1, 2))

    # rnn/concat
    # concat_crop_pool = tf.get_default_graph().get_tensor_by_name("concat_1:0").eval()
    # vals['concat_crop_pool'] = np.transpose(concat_crop_pool, (0, 3, 1, 2))
    #
    # concat_crop_pool_flat = tf.get_default_graph().get_tensor_by_name("Reshape_3:0").eval()
    # vals['conv_flat'] = concat_crop_pool_flat

    hidden_conv_concat = tf.get_default_graph().get_tensor_by_name("concat_1:0").eval()
    vals['hidden_conv_concat'] = hidden_conv_concat

    hidden_to_hidden = tf.get_default_graph().get_tensor_by_name("FC/xw_plus_b:0").eval()
    vals['hidden_to_hidden'] = hidden_to_hidden

    tanH = tf.get_default_graph().get_tensor_by_name("FC/Tanh:0").eval()
    vals['tanh_activation'] = tanH

    prediction = tf.get_default_graph().get_tensor_by_name("FC_1/xw_plus_b:0").eval()
    vals['prediction'] = prediction
    with open(dump_dir + '/tf_activations.p','wb') as f:
        pickle.dump(vals, f)


########################################################################################################################
def dump_tf_weights(dump_dir):
    trained_vars = get_model_weights()
    with open(dump_dir + '/trained_vars.p','wb') as f:
        pickle.dump(trained_vars, f)





########################################################################################################################

def _eval_once(saver, summary_writer, rmse_op, summary_op, avg_pred, filenames, preds, image_pyramids, init_shape_pyramids):
    """Runs Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      rmse_op: rmse_op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:

        saver.restore(sess, FLAGS.checkpoint_file)

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            num_iter = int(math.ceil(FLAGS.num_examples / int(filenames.get_shape()[0])))
            # Counts the number of correct predictions.
            errors = []
            predictions = []
            fNames = []
            gt = []

            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            if FLAGS.dump_tf_weights:
                dump_dir = '/home/gilsh/tmp/'
                dump_tf_weights(dump_dir)
                dump_tf_activations(dump_dir)


            print('%s: starting evaluation' % datetime.now())
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                # filename, GTlms, pred, rmse, pred0 = sess.run(
                #     [filenames, avg_pred, rmse_op, preds['0']])
                summary_str, image_pyramid_level0, init_shape_pyramid_level0, filename, pred, rmse, pred0, error_est = sess.run(
                    [summary_op, image_pyramids[0], init_shape_pyramids[0], filenames, avg_pred, rmse_op, preds['0'],
                     error_estimation_0])


                fNames.append(filename)
                predictions.append(pred)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                # gt.append(GTlms)

                errors.append(rmse)

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = FLAGS.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            if FLAGS.writeResults:
                if not os.path.isdir(FLAGS.resDir):
                    os.mkdir(FLAGS.resDir)
                for sIter in range(step):
                    for wIter in range(pred.shape[0]):
                        # resPath = FLAGS.resDir + os.path.basename(fNames[sIter][wIter])[:-4] + '.pts'
                        dir_path = FLAGS.resDir + os.path.dirname(fNames[sIter][wIter])
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        output_path = FLAGS.resDir + fNames[sIter][wIter][:-4] + '.pts'
                        print ('writing ' + output_path)
                        export_landmark_file(menpo.shape.PointCloud(predictions[sIter][wIter]), output_path, extension=None,
                                             overwrite=True)
                        error_prediction_file_path = FLAGS.resDir + fNames[sIter][wIter][:-4] + '.txt'
                        with open(error_prediction_file_path,'w') as f:
                            f.write(str(error_est[wIter]))

                        if FLAGS.writeGTlms:
                            resPath = FLAGS.GTlmsDir + os.path.basename(fNames[sIter][wIter])[:-4] + '.pts'
                            export_landmark_file(menpo.shape.PointCloud(gt[sIter][wIter]), resPath, extension=None,
                                                 overwrite=True)
                print('files written.')
                # FLAGS.writeResults = False

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def flip_predictions(predictions, shapes):
    flipped_preds = []

    for pred, shape in zip(predictions, shapes):
        pred = menpo.shape.PointCloud(pred)
        pred = utils.mirror_landmarks_68(pred, shape)
        flipped_preds.append(pred.points)

    return np.array(flipped_preds, np.float32)


def evaluate(dataset_path=''):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device(FLAGS.eval_device):
        #    train_dir = Path(FLAGS.checkpoint_dir)
        ref_dir = os.path.relpath(FLAGS.reference_shape_path)

        # gil - no need for init by pose
        initial_points_locations = mio.import_pickle(ref_dir + '/reference_shape.pkl')
        initial_points_subset = initial_points_locations[common_params.FLM_INDEXES,:]

        reference_shape = np.stack([initial_points_subset,
                                    initial_points_subset,
                                    initial_points_subset], axis=0)


        # reference_shape = mio.import_pickle(ref_dir + '/reference_shapepkl')
        with open(FLAGS.eval_filenames, 'rt') as f:
            filenames = [name.strip() for name in f.readlines()]

        margin = common_params.MARGIN
        filenames, image_pyramids, init_shape_pyramids, Tmats = data_provider.batch_inputs_pyr(
             filenames, FLAGS.data_dir, FLAGS.gt_path, FLAGS.bb_root, '', '', '',
             reference_shape, roll_support=False, grayscale=grayscale, num_landmarks=common_params.FLM_COUNT, margin=margin)

        # image_shapes = get_pyramid_shapes('/raid/algo/SOCVISION_SLOW/FLM/flm_db/images/test_set/0012.jpg')
        # mirrored_images, _, mirrored_inits, shapes, _, _, _ = data_provider.batch_inputs(
        #     [dataset_path], reference_shape,
        #     batch_size=FLAGS.batch_size, is_training=False, num_landmarks=FLAGS.num_of_flms, mirror_image=True, bb_root_input=FLAGS.bb_root, pose_root_input=FLAGS.pose_root, prev_frame_init = FLAGS.prev_frame_init)

        print('Loading model...')
        # Build a Graph that computes the logits predictions from the
        # inference model.

        # gcn1 = GCN.GCN(reference_shape[0], common_params.PATCH_INDEXES, FLAGS.batch_size)
        # gcn2 = GCN.GCN(reference_shape[0], common_params.PATCH_INDEXES, FLAGS.batch_size)
        #
        # gcn = [gcn1, gcn2]

        with tf.device(FLAGS.eval_device):
            patch_shape = (FLAGS.patch_size, FLAGS.patch_size)
            eval_start_time = time.time()
            #indices = mdm_model.get_patch_indices_stride2()
            #_, _, _, preds, _ = mdm_model.model_conv3_for101_less_patches_new(images, inits, indices,
            #                                                                            num_flms=FLAGS.num_of_flms)

            _, _, endpoints, preds, _ = mdm_model.model_conv3_pyramids(image_pyramids,
                                                               init_shape_pyramids,
                                                               common_params.PATCH_INDEXES,
                                                               num_iterations=common_params.ITERATIONS_COUNT,
                                                               patch_shape=(FLAGS.patch_size,FLAGS.patch_size),
                                                               num_flms= len(common_params.FLM_INDEXES),
                                                               num_channels=3,
                                                                       dropout=False
                                                                       )

            eval_time = time.time() - eval_start_time
            print('eval time = %.3f secs' % eval_time)
            # pred, _, _, preds, _ = mdm_model.model_sep_iter(images, inits, patch_shape=patch_shape)
            iter_str = FLAGS.get_output_of_iteration
            tf.get_variable_scope().reuse_variables()

#        pred_rolled = tf.py_func(roll_back, [preds[iter_str], images, rolls], [tf.float32])
#        avg_pred, _ = tf.py_func(apply_pts_transform, [pred_rolled[0], bbs_dst, bbs_src], [tf.float32, tf.int64])
        avg_pred, _ = tf.py_func(mdm_eval_utils.apply_pts_transform_new, [preds[iter_str], Tmats],
                                 [tf.float32, tf.int64])

        global error_estimation_0
        error_estimation_0 = endpoints['error_estimation_1']
        DEBUG = 0
        if DEBUG:
            image = np.zeros((255,255,3))
            p0 = np.zeros((1, 165,2))
            p1 = np.zeros((1, 165,2))
            out = mdm_eval_utils.render_points_on_image(image, p0, p1, filenames)

        if WRITE_TB_DATA:
            debug_l0 = tf.py_func(mdm_eval_utils.render_points_on_image, [image_pyramids[2], endpoints['init_0'], endpoints['final_0'], filenames],[tf.float32])
            debug_l0[0].set_shape((1,None, None, 3))
            debug_l1 = tf.py_func(mdm_eval_utils.render_points_on_image, [image_pyramids[1], endpoints['init_1'], endpoints['final_1'],  filenames], [tf.float32])
            debug_l1[0].set_shape((1,None, None, 3))
            debug_l2 = tf.py_func(mdm_eval_utils.render_points_on_image, [image_pyramids[0], endpoints['init_2'], endpoints['final_2'],  filenames], [tf.float32])
            debug_l2[0].set_shape((1,None, None, 3))
            # debug_l3 = tf.py_func(mdm_eval_utils.render_points_on_image, [image_pyramids[0], endpoints['init_3'], endpoints['final_3']], [tf.float32])
            # debug_l3[0].set_shape((1,None, None, 3))

        # Calculate predictions.
        # norm_error = mdm_model.normalized_rmse(avg_pred, gt_truth, FLAGS.num_of_flms)
        norm_error = mdm_model.normalized_rmse(avg_pred, avg_pred, common_params.FLM_COUNT)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            mdm_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        # tf.train.image("image", image_pyramids[1])
        # summary_op = tf.merge_summary(summaries)

        # pred_images, = tf.py_func(utils.batch_draw_landmarks,
        #
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, '')
        if WRITE_TB_DATA:
            summary0 = tf.image_summary('pyr_0', debug_l0[0],1)
            summary1 = tf.image_summary('pyr_1', debug_l1[0],1)
            summary2 = tf.image_summary('pyr_2', debug_l2[0],1)
            # summary3 = tf.image_summary('pyr_3', debug_l3[0],1)

            summaries.append(summary0)
            summaries.append(summary1)
            summaries.append(summary2)
            # summaries.append(summary3)

        summary_op = tf.merge_summary(summaries)

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.tensorboard_dir,
                                                graph_def=graph_def)
        # summary_op = []
        # summary_writer = []

        # data_provider.load_image('/root/sharedfolder/flm/datasets/300W_LP_semi_frontal/AFW/AFW_5201079592_1_8.jpg',
        #                          mio.import_pickle(train_dir / 'reference_shape.pkl'), is_training=False, group='PTS',
        #                          mirror_image=False, bb_root_input='/root/sharedfolder/flm/mdm/mdm/bbs/AFW')

        # while True:
        for model_step in model_steps:
            model = 'model.ckpt-%d' % model_step
            # FLAGS.checkpoint_file = FLAGS.model_dir + model
            # FLAGS.resDir = experiment_dir + '/res_%d/' % model_step
            _eval_once(saver, summary_writer, norm_error, summary_op, avg_pred, filenames, preds, image_pyramids, init_shape_pyramids)
            # if FLAGS.run_once:
            #  break
            # time.sleep(FLAGS.eval_interval_secs)

def roll_back(batch_pts, images, rolls, verbose = 0):
    if verbose:
       print('roll = '+str(rolls))
       print('pts = ' + str(batch_pts.shape))

    rolled_pts = []
    batch_size = batch_pts.shape[0]
    for i in range(batch_size):
        pts = batch_pts[i]
        im_shape = images[i].shape
        h, w = im_shape[:2]
        if verbose:
            print('im shape for roll back = ' + str(h) + ',' + str(w))

        rotation_center = (w / 2.0, h / 2.0)
        #rotation_center = data_provider.calc_bb_center(bbs_dst, MARGIN_DOWN, MARGIN_UP)
        #rotation_center = (rotation_center[1], rotation_center[0])
        points_in_image_center_system = np.array([(x, y, 1) for (y, x) in pts])
        inv_points_mat = np.transpose(points_in_image_center_system)
        rot_mat = cv2.getRotationMatrix2D(rotation_center, rolls[i], 1)

        rotated_points = rot_mat.dot(inv_points_mat)
        final_points = np.transpose(rotated_points)

        pts = np.array([(x, y) for (y, x) in final_points])
        rolled_pts.append(pts.astype('float32'))

    rolled_pts = np.array(rolled_pts)
#    print('pts = ' + str(rolled_pts.shape))
    return rolled_pts

def apply_pts_transform(pts, bbs, bbs_n):
    # print(bbs_n.shape)

    # no transform
    pts_return = pts.copy()
    batch_size = pts.shape[0]
    for i in range(batch_size):
        factor_x = float(bbs_n[i, 2] - bbs_n[i, 0] + 1) / (bbs[i, 2] - bbs[i, 0] + 1)
        factor_y = float(bbs_n[i, 3] - bbs_n[i, 1] + 1) / (bbs[i, 3] - bbs[i, 1] + 1)
        pts_return[i, :, 1] = (pts[i, :, 1] - bbs[i, 0]) * factor_x + bbs_n[i, 0]
        pts_return[i, :, 0] = (pts[i, :, 0] - bbs[i, 1]) * factor_y + bbs_n[i, 1]

    # no transform end

    # for j in range(10):
    #     for i in range(35):
    #         pts_return[j, i, 1] = (pts[j, i, 1] - bbs[0]) * factor_x + bbs_n[0]
    #         pts_return[j, i, 0] = (pts[j, i, 0] - bbs[1]) * factor_y + bbs_n[1]
    # pts_return[:, :, 1] = (pts[:, :, 1] - bbs[0]) * factor_x + bbs_n[0]
    # pts_return[:, :, 0] = (pts[:, :, 0] - bbs[1]) * factor_y + bbs_n[1]
    return pts_return, 1


if __name__ == '__main__':
    evaluate()

