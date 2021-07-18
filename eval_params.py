

import tensorflow as tf
import os



# num_flms = 165
fd = 'hfd'
mode = 'sil'
grayscale = False
num_channels = 1 if grayscale else 3
# num_patches = 165
# desc = '_'.join([str(num_flms), fd, mode])
# full_desc = desc
patched_indices_func = ''


FIGURE_PATH = 'recentResults/lastFig.png'


mode = 'silhouette' # '3d' or 'silhouette'


reference_shape = '/opt/nwtc/WFLW/'
model_dir = '/opt/nwtc/models/49_p_l1_loss_wflw/'

gt_dir = ''
eval_files = '/opt/nwtc/WFLW/wflw_expressions.txt'

image_dir = '/opt/nwtc/WFLW/test/all/images/'
bbs_dir = '/opt/nwtc/WFLW/bb/test/all/images/'


with open(eval_files) as f:
    for i, l in enumerate(f):
        pass

file_count = i + 1

model_steps = [21300]  # range(300, 1200, 300)


results_dir = '/opt/nwtc/models/49_p_l1_loss_wflw/res_21300/wflw_expressions/'
model = 'model.ckpt-%d' % model_steps[0]
dump_tf_weights = False

tensorboard_dir = '/tmp/'
pretrained_model = model_dir + model
get_output_of_iteration = '2'


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('get_output_of_iteration', get_output_of_iteration,
                           """the iteration to take the output from.""")


tf.app.flags.DEFINE_boolean('dump_tf_weights', dump_tf_weights,
                           """To dump or not to dump this is the question.""")

tf.app.flags.DEFINE_string('model_dir', mode,
                           """Directory where tf checkpoints are located.""")

tf.app.flags.DEFINE_string('checkpoint_file', pretrained_model,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('reference_shape_path', reference_shape,
                           '''Reference shapes location.''')
tf.app.flags.DEFINE_integer('eval_interval_secs', 120 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', file_count,
                            """Number of examples to run.""")

tf.app.flags.DEFINE_string('eval_filenames', eval_files,
                           """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_string('device', '/cpu:0', 'the device to eval on.')

tf.app.flags.DEFINE_string('resDir', results_dir,
                           """Directory where to write results.""")
tf.app.flags.DEFINE_string('writeResults', True,
                           """Whether results should be written""")

tf.app.flags.DEFINE_string('writeGTlms', False,
                           """Whether results should be written""")
tf.app.flags.DEFINE_string('eval_device', '/cpu:0', """Device to eval with.""")

tf.app.flags.DEFINE_string('tensorboard_dir', tensorboard_dir, """Tensorboard dir""")

tf.app.flags.DEFINE_string('data_dir', image_dir,
                           """Directory of the images and matching landmarks.""")
tf.app.flags.DEFINE_string('gt_path', gt_dir,
                           """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_string('bb_root', bbs_dir,
                           """Directory where the bbs are.""")
tf.app.flags.DEFINE_integer('with_pose', 1, 'whether to train with pose - use only in vra')


tf.app.flags.DEFINE_string('face_detector', 'NFD', """Face detector (NFD/HFD)""")



WRITE_TB_DATA = False
FLAGS.num_preprocess_threads = 1
FLAGS.prev_frame_init = False
FLAGS.patch_size = 14
FLAGS.batch_size = 7


