
import os
import cv2
import shutil
from show_flm_on_image import read_pts

def compute_pts_bb(pts_file_name):
    x_points, y_points = read_pts(pts_file_name)
    min_x = min(x_points)
    max_x = max(x_points)
    min_y = min(y_points)
    max_y = max(y_points)
    return min_x, min_y, max_x - min_x, max_y - min_y





def add_dir_to_aggregate(orig_db_dir, file_prefix, aggregate_db_dir):
    for filename in os.listdir(orig_db_dir):
        new_file_name = file_prefix + filename
        #if this is an image - make sure width % 4  = 0 (for VRA)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = cv2.imread(orig_db_dir + filename)
            image_width = image.shape[1]
            image_height = image.shape[0]

            if image_width % 4 != 0:
                image_width = image_width & 0xfffffff4
                image = image[0:image_height, 0:image_width]

            cv2.imwrite(aggregate_db_dir + "\\" + new_file_name[0:len(new_file_name) - 3] + "jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif filename.endswith(".t7"):
            t7_file_name = orig_db_dir + filename
            pts_file_name = aggregate_db_dir + "\\" + new_file_name[0:len(new_file_name) - 2] + "pts"
            translate_t7_to_pts(t7_file_name, pts_file_name)
            upper_left_x, upper_left_y, width, height = compute_pts_bb(pts_file_name)
            # create output file for VRA: (only once per face)
            param_file_name = aggregate_db_dir + "\\" + new_file_name[0:len(new_file_name) - 3] + "jpg.txt"
            file = open(param_file_name, 'w')
            file.write("%d,%d,1\n" % (image_width, image_height))
            file.write("%d,%d,%d,%d\n" % (upper_left_x, upper_left_y, width, height))
            file.close()

        elif filename.endswith(".pts"):
            shutil.copyfile(orig_db_dir + filename, aggregate_db_dir + "\\" + new_file_name)
        #
        # if filename.endswith(".pts"):
            # create output file for VRA: (only once per face)
            upper_left_x, upper_left_y, width, height = compute_pts_bb(orig_db_dir + filename)
            param_file_name = aggregate_db_dir + "\\" + new_file_name[0:len(new_file_name) - 3] + "jpg.txt"
            file = open(param_file_name, 'w')
            file.write("%d,%d,1\n" % (image_width, image_height))
            file.write("%d,%d,%d,%d\n" % (upper_left_x, upper_left_y, width, height))
            file.close()


def create_aggregate_db(orig_db_dir, aggregate_db_dir, skip_dirs):

    #recursive traverse the source db root
    for root, dirs, files in os.walk(orig_db_dir):
        if len(files) == 0:
            continue
        if root in skip_dirs:
            continue
        diff_path = root[len(orig_db_dir) + 1:]
        file_prefix = diff_path.replace("\\", "_") + "_"
        should_skip = 0
        for d in skip_dirs:
            index = file_prefix.find(d)
            if index != -1:
                should_skip = 1
                break
        if should_skip == 1:
            continue

        add_dir_to_aggregate(root + "\\", file_prefix, aggregate_db_dir)


if __name__ == "__main__":

    orig_db_dir = "C:\\work\\FLM\\DB\\original"
    aggregate_db_dir = "C:\\work\\FLM\\DB\\with_2d_fan\\all_data_flat"
    skip_dirs = ["CatA", "CatB", "CatC"]
    create_aggregate_db(orig_db_dir, aggregate_db_dir, skip_dirs)
