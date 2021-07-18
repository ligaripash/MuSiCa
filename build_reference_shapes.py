import data_provider

base_dir = '/root/shared_data/300W_LP/semi_frontal/'
out_dir = base_dir + 'experiments/reference_shapes_101/'
gt_dir = base_dir + '300W_LP_semi_frontal_101/'
bb_dir = base_dir + '300W_LP_semi_frontal_99/300W_LP_semi_frontal_bbs_VRA_new/'
pose_dir = base_dir + '300W_LP_semi_frontal_99/300W_LP_semi_frontal_pose_VRA_new/'
files_file = base_dir + 'all_filenames1.txt'

with open(files_file,'rt') as f:
    filenames = f.readlines()
filenames = [ff.strip() for ff in filenames]
refs = data_provider.build_reference_shape_per_pose_by_files(filenames, gt_dir, bb_dir, pose_dir, 3)
data_provider.save_reference_shape(refs, 1, out_dir)

