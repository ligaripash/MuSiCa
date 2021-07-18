"""A library to evaluate MDM on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path

import data_provider
import menpo
import matplotlib
import numpy as np
import tensorflow as tf
import utils
import cv2


########################################################################################################################

def render_points_on_image(image, point_set_a, point_set_b, filename):
    # gil
    # print (image.shape)
    out_image = np.array(image[0,:,:,:]*255)
    # out_image = np.reshape(out_image, (1, out_image.shape[0], out_image.shape[1], out_image.shape[2]))
    # return out_image
    # print ('shape', str(out_image.shape))
    f_name = '/mnt/home/tmp/' + filename[0].split('/')[1][:-3] + str(out_image.shape[0]) + 'x' + str(out_image.shape[1]) + '.jpg'
    print (f_name)
    for point in point_set_a[0,:,:]:
        # print (int(point[1]), int(point[0]))
        # out_image[int(point[0]), int(point[1])] = (255,0,0)
        try:
            out_image[int(point[0]), int(point[1])] = (255,0,0)
        except:
            continue

        # out_image = cv2.circle(out_image, (int(point[1]), int(point[0])), 1, (255,0,0))

        # out_image[(int(point[1]), int(point[0]))] = cv2.circle(out_image, (int(point[1]), int(point[0])), 2, (255, 0, 0))


    for point in point_set_b[0,:,:]:
        # out_image[int(point[0]), int(point[1])] = (0, 255, 0)
        try:
            out_image[int(point[0]), int(point[1])] = (0,255,0)
        except:
            continue

        # out_image = cv2.circle(out_image, (int(point[1]), int(point[0])), 1, (0,255,0))

    cv2.imwrite(f_name, out_image)
    new_out = np.reshape(out_image, (1,out_image.shape[0], out_image.shape[1], out_image.shape[2]))
    return new_out

########################################################################################################################

def plot_ced(errors, method_names=['MDM']):
    from matplotlib import pyplot as plt
    from menpofit.visualize import plot_cumulative_error_distribution
    import numpy as np
    # plot the ced and store it at the root.
    fig = plt.figure()
    fig.add_subplot(111)
    plot_cumulative_error_distribution(errors, legend_entries=method_names,
                                       error_range=(0, 0.09, 0.005))
    # shift the main graph to make room for the legend
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

def apply_pts_transform_new(batch_pts, Tmats):
    pts_return = []
    batch_size = batch_pts.shape[0]
    for i in range(batch_size):
        pts = batch_pts[i]
        inv_T = np.linalg.inv(Tmats[i])
        transformed_pts = data_provider.transform_points(pts, inv_T)
        pts_return.append(transformed_pts)

    pts_return = np.array(pts_return).astype('float32')

    return pts_return, 1


def flip_predictions(predictions, shapes):
    flipped_preds = []

    for pred, shape in zip(predictions, shapes):
        pred = menpo.shape.PointCloud(pred)
        pred = utils.mirror_landmarks_68(pred, shape)
        flipped_preds.append(pred.points)

    return np.array(flipped_preds, np.float32)

