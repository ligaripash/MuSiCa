import tensorflow as tf
import tensorflow as tf
import os
import common_params

base_dir = '/opt/nwtc'
work_dir = '/opt/nwtc/models/49_p_l1_loss_wflw/'
model_dir = '/opt/nwtc/models/49_p_l1_loss_wflw/'
gt_dir = '/opt/nwtc/WFLW/train/images/'
image_dir = '/opt/nwtc/WFLW/train/images/'
pose_dir = ''
roll_dir = ''
bbs_dir = '/opt/nwtc/WFLW/bb/train/images/'
train_file_list = '/opt/nwtc/WFLW/wflw_files_augmented_mahal.txt'
init_shape_path = '/opt/nwtc/WFLW/'
is_finetune = 0
model_step = ''
pretrained_model = ''
face_detector = 'NFD'
MOVING_AVERAGE_DECAY = 0.9999
SCALE_AUG_FACTOR = 10.0 # percent out of the bounding box width
TRANSLATION_AUG_FACTOR = 10.0 # percent out of the bounding box width
WRITE_TB_DATA = True


params = {
  "base_dir":base_dir,
  "work_dir":work_dir,
  "model_dir":model_dir,
  "gt_dir": gt_dir,
  "image_dir":   image_dir,
  "pose_dir": pose_dir,
  "roll_dir":   roll_dir,
  "bbs_dir": bbs_dir,
  "train_file_list": train_file_list,
  "init_shape_path": init_shape_path,
  "face_detector": face_detector,
  "MOVING_AVERAGE_DECAY":   MOVING_AVERAGE_DECAY,
  "SCALE_AUG_FACTOR":   SCALE_AUG_FACTOR,
  "TRANSLATION_AUG_FACTOR":   TRANSLATION_AUG_FACTOR,
  "MARGIN": common_params.MARGIN
}


if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


FLAGS = tf.app.flags.FLAGS

# train hyper parameters
########################################################################################################################
tf.app.flags.DEFINE_integer('is_finetune', is_finetune, """Is finetune.""")
tf.app.flags.DEFINE_string('optimizer', 'ADAM', """Optimizer Type""")
tf.app.flags.DEFINE_string('loss', 'L1', """loss type""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
#tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005,
                          """Initial learning rate - default 0.001 .""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 2,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('patch_size', 14, 'The extracted patch size')
tf.app.flags.DEFINE_integer('with_pose', not common_params.SINGLE_POSE_ONLY, 'whether to train with pose - use only in vra')
#tf.app.flags.DEFINE_integer('num_of_flms', 10, 'number of facial landmarks')
tf.app.flags.DEFINE_integer('num_epochs', 30000, 'number of  epochs over the train set')

# tf.app.flags.DEFINE_integer('margin_x', 0.3, 'horizontal margin ratio')
# tf.app.flags.DEFINE_integer('margin_down', 0.3, 'margin ratio below bb')
# tf.app.flags.DEFINE_integer('margin_up', 0.3, 'margin ratio above bb')
tf.app.flags.DEFINE_integer('margin', 0.3, 'margin ratio')

########################################################################################################################
# paths
tf.app.flags.DEFINE_string('face_detector', face_detector, """Face detector (NFD/HFD)""")
tf.app.flags.DEFINE_string('yaw_split',[-60, -15,15, 60],'''leftmost to rightmost and edge degrees in increasing order''')

tf.app.flags.DEFINE_string('train_dir', model_dir,
                           """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_string('image_dir', image_dir,
                           """Directory of the images.""")

tf.app.flags.DEFINE_string('train_filenames', train_file_list,
                           """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_string('gt_path', gt_dir,
                           """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_string('bb_root', bbs_dir,
                           """Directory where the bbs are.""")

tf.app.flags.DEFINE_string('pose_root', pose_dir,
                           """Directory where the poses are.""")

tf.app.flags.DEFINE_string('reference_shape_path',init_shape_path ,'''Reference shapes location.''')
# eval args
tf.app.flags.DEFINE_string('eval_filenames', train_file_list,
                           """Directory where to write event logs and checkpoint.""")
# pretrained model
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', pretrained_model,#work_dir + 'train_101_from_best99/model.ckpt',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_string('pretrained_model_step', model_step,
                           """If specified, restore this pretrained model step """
                           """before beginning any training.""")

tf.app.flags.DEFINE_string('weighted_loss', False,
                           """If specified, restore this pretrained model step """
                           """before beginning any training.""")

# os
# tf.app.flags.DEFINE_string('train_device', '/cpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")


