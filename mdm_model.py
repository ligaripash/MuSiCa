from menpo.shape.pointcloud import PointCloud
import slim
import tensorflow as tf
import utils
import numpy as np

from slim import ops
from slim import scopes


def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio


########################################################################################################################

def L1_loss(pred, gt_truth, num_of_flms=165):
    if num_of_flms == 68:
        leftTipIndex = 36
        rightTipIndex = 45
    elif num_of_flms == 39:
        leftTipIndex = 21
        rightTipIndex = 30
    elif num_of_flms == 35:
        leftTipIndex = 6
        rightTipIndex = 12
    elif num_of_flms == 99 or num_of_flms == 101 or num_of_flms == 165:
        leftTipIndex = 55
        rightTipIndex = 73
    elif num_of_flms == 98:
        leftTipIndex = 60
        rightTipIndex = 72

    interoccular_dist = tf.reduce_sum((tf.abs(gt_truth[:, leftTipIndex, :] - gt_truth[:, rightTipIndex, :])), 1)

    errors = tf.reduce_sum(tf.abs(pred - gt_truth), 2)
    epsilon = 0.01
    comparison = tf.cast(tf.greater_equal(errors, tf.constant(epsilon)), tf.float32)
    rect_errors = tf.mul(errors, comparison)

    return tf.reduce_sum(rect_errors, 1) / (interoccular_dist * num_of_flms)


########################################################################################################################

def normalized_mse(pred, gt_truth, num_of_flms=165):
    if num_of_flms == 68:
        leftTipIndex = 36
        rightTipIndex = 45
    elif num_of_flms == 39:
        leftTipIndex = 21
        rightTipIndex = 30
    elif num_of_flms == 35:
        leftTipIndex = 6
        rightTipIndex = 12
    elif num_of_flms == 99 or num_of_flms == 101 or num_of_flms == 165:
        leftTipIndex = 55
        rightTipIndex = 73

    interoccular_dist_sqr = tf.reduce_sum(((gt_truth[:, leftTipIndex, :] - gt_truth[:, rightTipIndex, :]) ** 2), 1)
    # norm_sqr = tf.reduce_sum(((gt_truth[:, leftTipIndex, :] - gt_truth[:, rightTipIndex, :]) ** 2), 1)

    return tf.reduce_sum(tf.reduce_sum(tf.square(pred - gt_truth), 2), 1) / (interoccular_dist_sqr * num_of_flms)


########################################################################################################################


def normalized_rmse(pred, gt_truth, num_of_flms=68):
    # norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))
    if num_of_flms == 68:
        leftTipIndex = 36
        rightTipIndex = 45
    elif num_of_flms == 39:
        leftTipIndex = 21
        rightTipIndex = 30
    elif num_of_flms == 35:
        leftTipIndex = 6
        rightTipIndex = 12
    elif num_of_flms == 99 or num_of_flms == 101 or num_of_flms == 165:
        leftTipIndex = 55
        rightTipIndex = 73
    elif num_of_flms == 10:
        leftTipIndex = 4
        rightTipIndex = 9
    elif num_of_flms == 98:
        leftTipIndex = 60
        rightTipIndex = 72

    # gil
    interoccular_dist = tf.sqrt(tf.reduce_sum(((gt_truth[:, leftTipIndex, :] - gt_truth[:, rightTipIndex, :]) ** 2), 1))
    # norm_sqr = tf.reduce_sum(((gt_truth[:, leftTipIndex, :] - gt_truth[:, rightTipIndex, :]) ** 2), 1)

    return tf.reduce_sum(tf.sqrt(1e-3 + tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (
                interoccular_dist * num_of_flms)
    # gil

    # return tf.reduce_sum(tf.reduce_sum(tf.square(pred - gt_truth),2),1) / (interoccular_dist * num_of_flms)
    # return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred[:,0:26,:] - gt_truth[:,0:26,:]), 2)), 1) / (norm * 26) + tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred[:,26:35,:] - gt_truth[:,26:35,:]), 2)), 1) / (norm * 9 * 10)


def normalized_rmse_weighted(pred, gt_truth, num_of_flms=68, weight=4):
    # norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))
    if num_of_flms == 68:
        leftTipIndex = 36
        rightTipIndex = 45
        jawStart = 0
        jawEnd = 16
        nonJawStart = 17
        nonJawEnd = 67
    elif num_of_flms == 39:
        leftTipIndex = 21
        rightTipIndex = 30
        jawStart = 0
        jawEnd = 8
        nonJawStart = 9
        nonJawEnd = 38
    elif num_of_flms == 35:
        leftTipIndex = 6
        jawStart = 26
        jawEnd = 34
        nonJawStart = 0
        nonJawEnd = 25
        rightTipIndex = 12
    elif num_of_flms == 99 or num_of_flms == 101:
        leftTipIndex = 55
        rightTipIndex = 73
        jawStart = 0
        jawEnd = 18
        nonJawStart = 19
        if num_of_flms == 99:
            nonJawEnd = 98
        else:
            nonJawEnd = 100

    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, leftTipIndex, :] - gt_truth[:, rightTipIndex, :]) ** 2), 1))

    # return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * num_of_flms)

    return calc_rmse(pred, gt_truth, nonJawStart, nonJawEnd + 1, norm) + calc_rmse(pred, gt_truth, jawStart, jawEnd + 1,
                                                                                   norm) / weight


def conv_model_conv3(inputs, is_training=True, scope='', id_str=''):
    # summaries or losses.
    net = {}

    with tf.op_scope([inputs], scope, 'mdm_conv' + id_str):
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                # net['conv_1'+id_str] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1'+id_str)
                # gil
                net['conv_1' + id_str] = ops.conv2d(inputs, 32, [3, 3])
                net['pool_1' + id_str] = ops.max_pool(net['conv_1' + id_str], [2, 2])

                # crop_size = net['pool_1' + id_str].get_shape().as_list()[1:3]
                # net['input_cropped'+id_str] = utils.get_central_crop(inputs, box=crop_size)
                # net['conv_1_cropped'+id_str] = utils.get_central_crop(net['conv_1'+id_str], box=crop_size)

                # net['pool_1_concat'+id_str] = tf.concat(3, [net['input_cropped'+id_str], net['conv_1_cropped'+id_str], net['pool_1'+id_str]])
                # net['conv_1x1'+id_str]= ops.conv2d(net['pool_1_concat'+id_str], 32, [1, 1], scope='conv_1x1'+id_str)
                # gil - remove scope
                # net['conv_2'+id_str] = ops.conv2d(net['pool_1'+id_str], 32, [3, 3], scope='conv_2'+id_str)
                net['conv_2' + id_str] = ops.conv2d(net['pool_1' + id_str], 32, [3, 3])
                net['pool_2' + id_str] = ops.max_pool(net['conv_2' + id_str], [2, 2])

                crop_size = net['pool_2' + id_str].get_shape().as_list()[1:3]
                net['conv_2_cropped' + id_str] = utils.get_central_crop(net['conv_2' + id_str], box=crop_size)
                # net['input_cropped_2' + id_str] = utils.get_central_crop(inputs, box=crop_size)
                # net['conv_1_cropped_2' + id_str] = utils.get_central_crop(net['conv_1' + id_str], box=crop_size)

                net['concat' + id_str] = tf.concat(3, [net['conv_2_cropped' + id_str], net['pool_2' + id_str]])

                patch_attention = True
                if patch_attention:
                    patches_count = int(net['concat' + id_str].get_shape()[0])
                    net['flatten' + id_str] = tf.reshape(net['concat' + id_str], (patches_count, -1))
                    feature_size_per_patch = int(net['flatten' + id_str].get_shape()[1])
                    net['weight_per_patch'] = slim.ops.fc(net['flatten' + id_str], feature_size_per_patch,
                                                          activation=tf.sigmoid)
                    net['mul_weight'] = tf.mul(net['weight_per_patch'], net['flatten' + id_str])
                # net['conv_1x1_2' + id_str] = ops.conv2d(net['concat' + id_str], 64, [1, 1], scope='conv_1x1_2' + id_str)

                # gil - remove scope
                # net['conv_3'+id_str] = ops.conv2d(net['concat'+id_str], 64, [2, 2], scope='conv_3'+id_str)
                # gil - experiment without conv_3
                # net['conv_3' + id_str] = ops.conv2d(net['concat' + id_str], 64, [2, 2])

    return net


def conv_model(inputs, is_training=True, scope='', id_str=''):
    # summaries or losses.
    net = {}

    with tf.op_scope([inputs], scope, 'mdm_conv' + id_str):
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                net['conv_1' + id_str] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1' + id_str)
                net['pool_1' + id_str] = ops.max_pool(net['conv_1' + id_str], [2, 2])

                # crop_size = net['pool_1' + id_str].get_shape().as_list()[1:3]
                # net['input_cropped'+id_str] = utils.get_central_crop(inputs, box=crop_size)
                # net['conv_1_cropped'+id_str] = utils.get_central_crop(net['conv_1'+id_str], box=crop_size)

                # net['pool_1_concat'+id_str] = tf.concat(3, [net['input_cropped'+id_str], net['conv_1_cropped'+id_str], net['pool_1'+id_str]])
                # net['conv_1x1'+id_str]= ops.conv2d(net['pool_1_concat'+id_str], 32, [1, 1], scope='conv_1x1'+id_str)

                net['conv_2' + id_str] = ops.conv2d(net['pool_1' + id_str], 32, [3, 3], scope='conv_2' + id_str)
                net['pool_2' + id_str] = ops.max_pool(net['conv_2' + id_str], [2, 2])

                crop_size = net['pool_2' + id_str].get_shape().as_list()[1:3]
                net['conv_2_cropped' + id_str] = utils.get_central_crop(net['conv_2' + id_str], box=crop_size)
                # net['input_cropped_2' + id_str] = utils.get_central_crop(inputs, box=crop_size)
                # net['conv_1_cropped_2' + id_str] = utils.get_central_crop(net['conv_1' + id_str], box=crop_size)

                net['concat' + id_str] = tf.concat(3, [net['conv_2_cropped' + id_str], net['pool_2' + id_str]])
                # net['conv_1x1_2' + id_str] = ops.conv2d(net['concat' + id_str], 64, [1, 1], scope='conv_1x1_2' + id_str)

                # net['conv_3'+id_str] = ops.conv2d(net['concat'+id_str], 128, [3, 3], scope='conv_3'+id_str)
    return net


def conv_model_patchsize_10(inputs, is_training=True, scope='', id_str=''):
    # summaries or losses.
    net = {}

    with tf.op_scope([inputs], scope, 'mdm_conv' + id_str):
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                net['conv_1' + id_str] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1' + id_str)
                net['pool_1' + id_str] = ops.max_pool(net['conv_1' + id_str], [2, 2])

                # crop_size = net['pool_1' + id_str].get_shape().as_list()[1:3]
                # net['input_cropped'+id_str] = utils.get_central_crop(inputs, box=crop_size)
                # net['conv_1_cropped'+id_str] = utils.get_central_crop(net['conv_1'+id_str], box=crop_size)

                # net['pool_1_concat'+id_str] = tf.concat(3, [net['input_cropped'+id_str], net['conv_1_cropped'+id_str], net['pool_1'+id_str]])
                # net['conv_1x1'+id_str]= ops.conv2d(net['pool_1_concat'+id_str], 32, [1, 1], scope='conv_1x1'+id_str)

                net['conv_2' + id_str] = ops.conv2d(net['pool_1' + id_str], 32, [3, 3], scope='conv_2' + id_str)
                # net['pool_2'+id_str] = ops.max_pool(net['conv_2'+id_str], [2, 2])

                crop_size = net['conv_2' + id_str].get_shape().as_list()[1:3]
                net['conv_2_cropped' + id_str] = utils.get_central_crop(net['pool_1' + id_str], box=crop_size)
                # net['input_cropped_2' + id_str] = utils.get_central_crop(inputs, box=crop_size)
                # net['conv_1_cropped_2' + id_str] = utils.get_central_crop(net['conv_1' + id_str], box=crop_size)

                net['concat' + id_str] = tf.concat(3, [net['conv_2_cropped' + id_str], net['conv_2' + id_str]])
                # net['conv_1x1_2' + id_str] = ops.conv2d(net['concat' + id_str], 64, [1, 1], scope='conv_1x1_2' + id_str)

                net['conv_3' + id_str] = ops.conv2d(net['concat' + id_str], 64, [2, 2], scope='conv_3' + id_str)

    return net


def model(images, inits, num_iterations=4, num_patches=68, patch_shape=(26, 26), num_channels=3):
    # batch_size = images.get_shape()[0]
    batch_size = images.get_shape().as_list()[0]
    hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = tf.image.extract_patches(images, tf.constant(patch_shape), inits + dx)
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        # A = [patches[:,i,:,:,:] for i in range(68)]
        # patches = tf.concat(0, A)
        # patches = tf.reshape(patches, (-1, patch_shape[0], patch_shape[1], num_channels))

        endpoints['patches'] = patches

        with tf.variable_scope('convnet', reuse=step > 0):
            net = conv_model(patches)
            ims = net['concat']

        # ims = tf.reshape(ims, (-1, num_patches*ims._shape[-1]._value*ims._shape[-2]._value*ims._shape[-3]._value))
        ims = tf.reshape(ims, (batch_size, -1))

        with tf.variable_scope('rnn', reuse=step > 0) as scope:
            hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
            prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)

    return inits + dx, dxs, endpoints


def model_new(images, inits, hidden_init=np.zeros(512), num_iterations=4, num_patches=68, patch_shape=(26, 26),
              num_channels=3):
    # batch_size = images.get_shape()[0]
    batch_size = images.get_shape().as_list()[0]
    hidden_init_batch = np.repeat(np.expand_dims(hidden_init, axis=0), batch_size, axis=0)
    hidden_state = tf.zeros((batch_size, 512)) + tf.convert_to_tensor(hidden_init_batch, dtype=np.float32)
    # hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    flm_prediction = {}
    hidden_states = {}

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = tf.image.extract_patches(images, tf.constant(patch_shape), inits + dx)
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        # A = [patches[:,i,:,:,:] for i in range(68)]
        # patches = tf.concat(0, A)
        # patches = tf.reshape(patches, (-1, patch_shape[0], patch_shape[1], num_channels))

        endpoints['patches'] = patches

        with tf.variable_scope('convnet', reuse=step > 0):
            net = conv_model(patches)
            ims = net['concat']

        # ims = tf.reshape(ims, (-1, num_patches*ims._shape[-1]._value*ims._shape[-2]._value*ims._shape[-3]._value))
        ims = tf.reshape(ims, (batch_size, -1))

        with tf.variable_scope('rnn', reuse=step > 0) as scope:
            hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
            prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)
        flm_prediction[str(step)] = inits + dx
        hidden_states[str(step)] = hidden_state

    endpoints['hidden_state'] = hidden_state

    return inits + dx, dxs, endpoints, flm_prediction, hidden_states


########################################################################################################################


def select_patches_dx(dx, patches_indexes):
    # dx has the shape (128,165,2). The patches indexes are from the second dimension
    dx_t = tf.transpose(dx, [1, 2, 0])
    selected_dx = tf.gather(dx_t, patches_indexes)
    # transpose back
    selected_dx_t = tf.transpose(selected_dx, [2, 0, 1])
    return selected_dx_t


########################################################################################################################

def model_conv3_pyramids(image_pyramids, init_shape_pyramids, patches_indexes, hidden_init=np.zeros(512),
                         num_iterations=4, patch_shape=(26, 26), num_channels=3, num_flms=165, dropout=False):
    # batch_size = images.get_shape()[0]
    num_patches = len(patches_indexes)
    batch_size = image_pyramids[0].get_shape().as_list()[0]
    hidden_init_batch = np.repeat(np.expand_dims(hidden_init, axis=0), batch_size, axis=0)
    hidden_state = tf.zeros((batch_size, 512)) + tf.convert_to_tensor(hidden_init_batch, dtype=np.float32)
    # hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_flms, 2))
    endpoints = {}
    dxs = []
    flm_prediction = {}
    hidden_states = {}
    # gil - experiment with two iteration at the smallest scale
    if num_iterations == 4:
        step_to_pyr_level = [2, 2, 1, 0]
        scale_factor_between_iterations = [1, 2, 2, 2]
        scale_factor_to_base_pyr_level = [4, 4, 2, 1]
    elif num_iterations == 3:
        step_to_pyr_level = [2, 1, 0]
        scale_factor_between_iterations = [2, 2, 2]
        scale_factor_to_base_pyr_level = [4, 2, 1]
    elif num_iterations == 2:
        step_to_pyr_level = [1, 0]
        scale_factor_between_iterations = [2, 1]
        scale_factor_to_base_pyr_level = [2, 1]

    for step in range(num_iterations):
        # pyr_level = num_iterations - step - 1
        pyr_level = step_to_pyr_level[step]
        with tf.device('/cpu:0'):
            patch_dx = select_patches_dx(dx, patches_indexes)
            init_shape = select_patches_dx(init_shape_pyramids[pyr_level], patches_indexes)
            patches = tf.image.extract_patches(image_pyramids[pyr_level], tf.constant(patch_shape),
                                               init_shape + patch_dx)
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))

        endpoints['patches'] = patches

        # gil - the patches in each iteration are in different scale and probably need to learn different features.
        # with tf.variable_scope('convnet', reuse=False):
        net = conv_model_conv3(patches)
        ims = net['mul_weight']
        # ims = net['concat']
        # ims = net['conv_3']

        # ims = tf.reshape(ims, (-1, num_patches*ims._shape[-1]._value*ims._shape[-2]._value*ims._shape[-3]._value))
        ims = tf.reshape(ims, (batch_size, -1))

        ims_plus_hidden = tf.concat(1, [ims, hidden_state])

        # gil - old attention:
        # if step == 0:
        #   # gil - predict the weighting according to the concatenated features
        #   weighting = slim.ops.fc(ims, int(ims._shape[1]),  activation=tf.sigmoid)
        #   # if dropout:
        #   #     weighting = tf.nn.dropout(weighting, 0.1)
        #   #multiply each input element by its weighting
        #   ims_plus_hidden = tf.mul(ims, weighting)

        # with tf.variable_scope('rnn', reuse=step>0) as scope:
        # with tf.variable_scope('rnn', reuse=False) as scope:
        # hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
        hidden_state = slim.ops.fc(ims_plus_hidden, 512, activation=tf.tanh)
        if dropout:
            hidden_state = tf.nn.dropout(hidden_state, 0.5)

            # prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
        # Extra 1 place for error prediction
        fc_out = slim.ops.fc(hidden_state, num_flms * 2 + 1, activation=None)
        prediction = tf.slice(fc_out, [0, 0], [batch_size, num_flms * 2])
        prediction = tf.reshape(prediction, (batch_size, num_flms, 2))
        error_estimation = tf.slice(fc_out, [0, num_flms * 2], [batch_size, 1])
        error_estimation = tf.reshape(error_estimation, [batch_size])

        endpoints['prediction'] = prediction
        endpoints['error_estimation_' + str(step)] = error_estimation
        endpoints['init_' + str(step)] = init_shape_pyramids[pyr_level] + dx
        endpoints['final_' + str(step)] = endpoints['init_' + str(step)] + prediction

        # dx contains the prediction for the current pyramid level
        dx += prediction
        prediction_for_next_iteration = dx * scale_factor_between_iterations[step]
        # We append the prediction in level 0
        prediction_level_0 = dx * (scale_factor_to_base_pyr_level[step])
        dxs.append(prediction_level_0)
        flm_prediction[str(step)] = init_shape_pyramids[0] + prediction_level_0
        hidden_states[str(step)] = hidden_state
        dx = prediction_for_next_iteration

    endpoints['hidden_state'] = hidden_state

    return init_shape_pyramids[0] + dx, dxs, endpoints, flm_prediction, hidden_states


########################################################################################################################

def model_conv3(images, inits, hidden_init=np.zeros(512), num_iterations=4, num_patches=68, patch_shape=(26, 26),
                num_channels=3):
    # batch_size = images.get_shape()[0]
    batch_size = images.get_shape().as_list()[0]
    hidden_init_batch = np.repeat(np.expand_dims(hidden_init, axis=0), batch_size, axis=0)
    hidden_state = tf.zeros((batch_size, 512)) + tf.convert_to_tensor(hidden_init_batch, dtype=np.float32)
    # hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    flm_prediction = {}
    hidden_states = {}

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = tf.image.extract_patches(images, tf.constant(patch_shape), inits + dx)
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        # A = [patches[:,i,:,:,:] for i in range(68)]
        # patches = tf.concat(0, A)
        # patches = tf.reshape(patches, (-1, patch_shape[0], patch_shape[1], num_channels))

        endpoints['patches'] = patches

        with tf.variable_scope('convnet', reuse=False):
            net = conv_model_conv3(patches)
            ims = net['conv_3']

        # ims = tf.reshape(ims, (-1, num_patches*ims._shape[-1]._value*ims._shape[-2]._value*ims._shape[-3]._value))
        ims = tf.reshape(ims, (batch_size, -1))

        with tf.variable_scope('rnn', reuse=step > 0) as scope:
            hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
            prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)
        flm_prediction[str(step)] = inits + dx
        hidden_states[str(step)] = hidden_state

    endpoints['hidden_state'] = hidden_state

    return inits + dx, dxs, endpoints, flm_prediction, hidden_states


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
        'other': (42, 54)
    }
    return indices


def get_patch_indices_stride2(num_flms=101):
    flm_groups = get_indices_per_flm_group()
    strided_groups = ['jaw', 'browL', 'browR', 'eyeL', 'eyeR', 'nose', 'mouth']
    other_groups = ['noseBridge', 'pupils']
    other_points = ['other']

    inds = []
    for g in strided_groups:
        b, e = flm_groups[g]
        inds.extend(range(b, e + 1, 2))
    for g in other_groups:
        b, e = flm_groups[g]
        inds.extend(range(b, e + 1))
    for g in other_points:
        inds.extend(flm_groups[g])
    return inds


def get_patch_indices_stride3(num_flms=101):
    flm_groups = get_indices_per_flm_group()
    strided_groups = ['jaw', 'browL', 'browR', 'eyeL', 'eyeR', 'nose', 'mouth']
    # stride = [3, 3, 3, 3, 3, 2, 3]
    other_groups = ['noseBridge', 'pupils']
    other_points = ['other']

    inds = []
    i = 0
    for g in strided_groups:
        b, e = flm_groups[g]
        inds.extend(range(b, e + 1, 3))
        i += 1
    for g in other_groups:
        b, e = flm_groups[g]
        inds.extend(range(b, e + 1))
    for g in other_points:
        inds.extend(flm_groups[g])
    return inds


def get_patch_indices_minimal(num_flms=101):
    return np.array([1, 6, 10, 14, 19, 20, 22, 25, 28, 32, 34, 36, 39, 44, 46, 49, 52,
                     54, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 94, 98, 100, 101]) - 1


def model_init_step(images, inits, hidden_init=np.zeros(512), num_iterations=4, num_patches=68, patch_shape=(26, 26),
                    num_channels=3):
    # batch_size = images.get_shape()[0]
    batch_size = images.get_shape().as_list()[0]
    hidden_init_batch = np.repeat(np.expand_dims(hidden_init, axis=0), batch_size, axis=0)
    hidden_state = tf.zeros((batch_size, 512)) + tf.convert_to_tensor(hidden_init_batch, dtype=np.float32)
    # hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    flm_prediction = {}
    hidden_states = {}

    net_init = {}
    scale_factor = 4
    images_small = tf.image.resize_images(images, tf.round(images.get_shape()[1] / scale_factor),
                                          tf.round(images.get_shape()[2] / scale_factor))
    with tf.variable_scope('convnet_init'):
        with tf.op_scope([images_small], '', 'mdm_conv_init'):
            with scopes.arg_scope([ops.conv2d, ops.fc], is_training=True):
                with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                    net_init['conv_1_init'] = ops.conv2d(images_small, 32, [3, 3], scope='conv_1_init')
                    net_init['pool_1_init'] = ops.max_pool(net_init['conv_1_init'], [2, 2])
                    net_init['conv_2_init'] = ops.conv2d(net_init['pool_1_init'], 64, [3, 3], scope='conv_2_init')
                    net_init['pool_2_init'] = ops.max_pool(net_init['conv_2_init'], [2, 2])
                    net_init['conv_3_init'] = ops.conv2d(net_init['pool_2_init'], 128, [3, 3], scope='conv_3_init')
                    net_init['pool_3_init'] = ops.max_pool(net_init['conv_3_init'], [2, 2])
                    net_init['conv_4_init'] = ops.conv2d(net_init['pool_3_init'], 128, [3, 3], scope='conv_4_init')
                    net_init['pool_4_init'] = ops.max_pool(net_init['conv_4_init'], [2, 2])
                    net_init['conv_5_init'] = ops.conv2d(net_init['pool_4_init'], 64, [1, 1], scope='conv_5_init')
                    net_init['conv_output_init'] = tf.reshape(net_init['conv_5_init'], (batch_size, -1))
                    net_init['fc_1_init'] = slim.ops.fc(net_init['conv_output_init'], 512, activation=tf.nn.relu)
                    net_init['fc_2_init'] = slim.ops.fc(net_init['fc_1_init'], 512, activation=tf.nn.relu)
                    net_init['fc_3_init'] = slim.ops.fc(net_init['fc_2_init'], num_patches * 2, activation=None)
                    prediction_init = tf.reshape(net_init['fc_3_init'], (batch_size, num_patches, 2))

    inits = prediction_init
    for step in range(num_iterations):
        # step_str = '_'+str(step)
        step_str = ''
        with tf.device('/cpu:0'):
            patches = tf.image.extract_patches(images, tf.constant(patch_shape), inits + dx)
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        # A = [patches[:,i,:,:,:] for i in range(68)]
        # patches = tf.concat(0, A)
        # patches = tf.reshape(patches, (-1, patch_shape[0], patch_shape[1], num_channels))

        endpoints['patches' + step_str] = patches

        with tf.variable_scope('convnet' + step_str, reuse=step > 0):
            if patch_shape[0] > 14:
                net = conv_model_conv3(patches, id_str=step_str)
                ims = net['conv_3' + step_str]
            else:
                net = conv_model(patches, id_str=step_str)
                ims = net['concat' + step_str]

        # ims = tf.reshape(ims, (-1, num_patches*ims._shape[-1]._value*ims._shape[-2]._value*ims._shape[-3]._value))
        ims = tf.reshape(ims, (batch_size, -1))

        with tf.variable_scope('rnn' + step_str, reuse=step > 0) as scope:
            hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
            prediction = slim.ops.fc(hidden_state, num_patches * 2, activation=None)
            endpoints['prediction' + step_str] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)
        flm_prediction[str(step)] = inits + dx
        hidden_states[str(step)] = hidden_state

    endpoints['hidden_state' + step_str] = hidden_state

    return inits + dx, dxs, endpoints, flm_prediction, hidden_states, prediction_init


def model_cnnPerLms(images, inits, num_iterations=4, num_patches=68, patch_shape=(26, 26), num_channels=3):
    batch_size = images.get_shape().as_list()[0]
    hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    initPath = 'ckpt/train_22/variables_from_step_27900/'

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = tf.image.extract_patches(images, tf.constant(patch_shape), inits + dx)
        patches_ = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        endpoints['patches'] = patches_

        ims = []
        patches_list = tf.split(1, num_patches, patches)
        for lmsIter in xrange(num_patches):
            id_str = '_' + str(lmsIter)
            with tf.variable_scope('convnet' + id_str, reuse=step > 0):
                net = conv_model_withInits(
                    tf.reshape(patches_list[lmsIter], (batch_size, patch_shape[0], patch_shape[1], num_channels)),
                    id_str=id_str, initPath=initPath)
                ims.append(tf.reshape(net['concat' + id_str], (
                batch_size, 1, net['concat' + id_str]._shape[-3]._value, net['concat' + id_str]._shape[-2]._value,
                net['concat' + id_str]._shape[-1]._value)))

        ims = tf.reshape(ims, (batch_size, -1))

        with tf.variable_scope('rnn', reuse=step > 0) as scope:
            hidden_state = slim.ops.fc_withInits(tf.concat(1, [ims, hidden_state]), 512,
                                                 weight_init=np.load(initPath + 'rnn_weights'),
                                                 bias_init=np.load(initPath + 'rnn_biases'), activation=tf.tanh)
            prediction = slim.ops.fc_withInits(hidden_state, num_patches * 2, scope='pred',
                                               weight_init=np.load(initPath + 'pred_weights'),
                                               bias_init=np.load(initPath + 'pred_biases'), activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)

    return inits + dx, dxs, endpoints


def conv_model_withInits(inputs, is_training=True, scope='', id_str='', initPath=''):
    # summaries or losses.
    net = {}

    with tf.op_scope([inputs], scope, 'mdm_conv' + id_str):
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                net['conv_1' + id_str] = ops.conv2d_withInits(inputs, 32, [3, 3],
                                                              kernel_init=np.load(initPath + 'conv_1_kernels'),
                                                              bias_init=np.load(initPath + 'conv_1_biases'),
                                                              scope='conv_1' + id_str)
                net['pool_1' + id_str] = ops.max_pool(net['conv_1' + id_str], [2, 2])
                net['conv_2' + id_str] = ops.conv2d_withInits(net['pool_1' + id_str], 32, [3, 3],
                                                              kernel_init=np.load(initPath + 'conv_2_kernels'),
                                                              bias_init=np.load(initPath + 'conv_2_biases'),
                                                              scope='conv_2' + id_str)
                net['pool_2' + id_str] = ops.max_pool(net['conv_2' + id_str], [2, 2])

                crop_size = net['pool_2' + id_str].get_shape().as_list()[1:3]
                net['conv_2_cropped' + id_str] = utils.get_central_crop(net['conv_2' + id_str], box=crop_size)

                net['concat' + id_str] = tf.concat(3, [net['conv_2_cropped' + id_str], net['pool_2' + id_str]])
    return net