
####### API for creating face rects using a face detector #######



import cv2
import os
# import dlib
import glob

from create_aggregate_db import compute_pts_bb




def read_rect_from_file(output_file_path):
    if output_file_path == '':
        return 0,0,0,0

    rect_file = open(output_file_path, 'r')

    rect_file.readline()
    rect_file.readline()
    rect_file.readline()
    (left, top) = [float(x) for x in rect_file.readline().split(' ')]
    rect_file.readline()
    (right, bottom) = [float(x) for x in rect_file.readline().split(' ')]
    rect_file.close()

    return left, top, right, bottom



def write_rect_to_file(output_file_path, left, top, right, bottom):

    output_file = open(output_file_path,'w')
    output_file.write("version: 1\n")
    output_file.write("n_points: 4\n")
    output_file.write("{\n")
    output_file.write("%.3f %.3f\n" % (left, top))
    output_file.write("%.3f %.3f\n" % (left, bottom))
    output_file.write("%.3f %.3f\n" % (right, bottom))
    output_file.write("%.3f %.3f\n" % (right, top))
    output_file.write("}\n")
    output_file.close()



def compute_rect_error(rect, gt_rect):
    gt_rect_right = gt_rect[0] + gt_rect[2]
    gt_rect_left = gt_rect[0]
    gt_rect_top = gt_rect[1]
    gt_rect_bottom = gt_rect_top + gt_rect[3]

    left = max(rect.left(), gt_rect_left)
    right = min(rect.right(), gt_rect_right)
    bottom = min(rect.bottom(), gt_rect_bottom)
    top = max(rect.top(), gt_rect_top)

    error = abs(left - gt_rect_left) + abs(right - gt_rect_right) + abs(top - gt_rect_top) + abs(bottom - gt_rect_bottom)

    return error




def compute_intersection_over_union(rect, gt_rect):

    r_l = rect[0]
    r_t = rect[1]
    r_r = rect[2]
    r_b = rect[3]

    g_l = gt_rect[0]
    g_t = gt_rect[1]
    g_r = gt_rect[2]
    g_b = gt_rect[3]

    left = max(r_l, g_l)
    top = max(r_t, g_t)
    right = min(r_r, g_r)
    bottom = min(r_b, g_b)

    intersection_area = 0
    if left < right and bottom > top:
        intersection_area = (right - left) * (bottom - top)

    rect_area = (r_r - r_l) * (r_b - r_t)
    gt_area = (g_r - g_l) * (g_b - g_t)

    iot = float(intersection_area) / (rect_area + gt_area - intersection_area)

    return iot



########################################################################################################################

# def choose_final_rect(fd_rects, gt_rect):
#     intersection_rect_area_max_size = 0
#     final_rect = []
#     for rect in fd_rects:
#         rect_intersection_area = compute_intersection_area(rect, gt_rect)
#         if rect_intersection_area > intersection_rect_area_max_size:
#             intersection_rect_area_max_size = rect_intersection_area
#             final_rect = rect
#
#     return final_rect

def choose_final_rect(fd_rects, gt_rect):
    min_error = 999999
    final_rect = []
    for rect in fd_rects:
        error = compute_rect_error(rect, gt_rect)
        if error < min_error:
            min_error = error
            final_rect = rect

    return final_rect


########################################################################################################################

def create_dlib_face_rect(input_dir, input_subset, output_dir, debug):
    face_detector = dlib.get_frontal_face_detector()

    for full_filename in glob.glob(input_dir + "\\" + input_subset):
        filename = os.path.basename(full_filename)
        output_file_path = output_dir + "\\" + filename
        if os.path.isfile(output_file_path): continue

        pts_file_name = input_dir + "\\" + filename
        gt_upper_left_x, gt_upper_left_y, gt_width, gt_height = compute_pts_bb(pts_file_name)
        gt_rect = (gt_upper_left_x, gt_upper_left_y, gt_width, gt_height)
        image_file = input_dir + "\\" + filename[0:len(filename) - 3] + "jpg"
        image = cv2.imread(image_file)
        rects, scores, idx = face_detector.run(image, 1, -1)
        final_rect = choose_final_rect(rects, gt_rect)

        if debug:
            p1 = (final_rect.left(), final_rect.top())
            p2 = (final_rect.right(), final_rect.bottom())
            cv2.rectangle(image, p1, p2, (0, 0, 255))
            cv2.imshow(image_file, image)
            key = cv2.waitKey(0) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        #write rect
        #output_file = open(output_file_path,'w')
        if final_rect != []:
            write_rect_to_file(output_file_path, final_rect.left(), final_rect.top(), final_rect.right(), final_rect.bottom())
            #output_file.write("%d,%d,%d,%d\n" % (final_rect.left(), final_rect.top(), final_rect.width(), final_rect.height()))
        else:
            print filename + " fd didnt find the gt face\n"
            write_rect_to_file(output_file_path, 0.0, 0.0, 0.0, 0.0)
            #output_file.write(
                #"%d,%d,%d,%d\n" % (0, 0, 0, 0))
        #output_file.close()















if __name__ == "__main__":

    #input_dir = "C:\\work\\FLM\\DB\\original\\helen\\testset"
    input_dir = "C:\\work\\FLM\\DB\\no_2d_fan\\all_data_flat"
    input_subset = "lfpw_testset_*.pts"

    output_dir = "C:\\work\\FLM\\rects\\dlib_rects"
    create_dlib_face_rect(input_dir, input_subset,output_dir, False)