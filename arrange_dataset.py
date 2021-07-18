import math
import os
from random import shuffle
import glob
import data_provider

#base_dir = '/mnt/Databases/FLM_Data/FLM_Data/300W_LP/semi_frontal/'
#base_dir = '/root/shared_data/300W_LP/semi_frontal/'
base_dir = '/raid/algo/SOCVISION_SLOW/FLM/300W_LP/semi_frontal/'
data_dir = base_dir + '300W_LP_semi_frontal_101_new/'
gt_dir = base_dir + '300W_LP_semi_frontal_165/'

#fd_dir = base_dir + 'NFD_v2_w_crop/'
#bb_root = fd_dir + 'bbs/'
#pose_root = fd_dir + 'yaw/'
bb_root = base_dir = '/raid/algo/SOCVISION/FLM/300W_LP/semi_frontal/300W_LP_semi_frontal_99/300W_LP_semi_frontal_bbs_VRA_new/'
bb_root = base_dir = '/raid/algo/SOCVISION/FLM/300W_LP/semi_frontal/300W_LP_semi_frontal_99/300W_LP_semi_frontal_bbs_VRA_new/'

datasets = ['AFLW','HELEN','LFPW','AFW']
all_ims = []

def write_files(outfile, filenames):
    f = open(outfile, 'wt')
    f.write('\n'.join(filenames))
    f.close()

def read_files(infile):
    f = open(infile, 'r')
    filenames = f.readlines()
    filenames = [name.strip() for name in filenames]
    f.close()
    return filenames

# no point in tr - val without subject marking
# for data in datasets:
#     print data
#     all_file = base_dir + 'files/' + data + '_nfd_v2b_filenames.txt'
#
#     all_ims_dataset = glob.glob(data_dir+data+ '/*.jpg') + glob.glob(data_dir+data+ '_Flip/*.jpg') #
#     print('all = ' + str(len(all_ims_dataset)))
#     all_ims_dataset = [os.path.relpath(f, data_dir) for f in all_ims_dataset]
#     all_ims_dataset = [f for f in all_ims_dataset if data_provider.is_nfd_file_ok(f, gt_dir, bb_root, pose_root)]
#     print('all ok = ' + str(len(all_ims_dataset)))
#     write_files(all_file, all_ims_dataset)
#     all_ims.extend(all_ims_dataset)

#shuffle(all_ims)
#print('all in all = '+str(len(all_ims)))
#write_files(base_dir+'files/all_v2b_filenames.txt', all_ims)
infile = base_dir+ 'files/nfd_w_crop.txt'
outfile = base_dir+ 'files/nfd_w_crop_filtered.txt'
f = open(outfile, 'wt')

filenames = read_files(infile)
print('There are %d files'%len(filenames))
status_dict = {}
i = 0
for name in filenames:
    if i%100 == 0:
        print(i)
    i += 1
    status = data_provider.is_nfd_file_ok(name, gt_dir, bb_root, pose_root)
    status_dict[status] = status_dict.get(status, 0) + 1
    if status == '':
        f.write('%s\n'%name)

print(status_dict)
f.close()




