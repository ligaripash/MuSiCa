
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
# import msvcrt as m
import glob
from matplotlib.pyplot import plot, draw, show
import matplotlib.image as mpimg
import numpy as np
import random
import difflib
import fnmatch


# from create_face_rect import read_rect_from_file

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





def compute_pts_bb(x_points, y_points):
    right = max(x_points)
    left = min(x_points)
    top = min(y_points)
    bottom = max(y_points)
    return left, top, right, bottom



def read_pts(filename):
    #open the pts file
    x_points = []
    y_points = []
    pts_file = open(filename,'r')
    pts_file.readline()
    line = pts_file.readline().split(" ")
    point_count = int(line[-1])
    #pts_file.readline()
    pts_file.readline()
    for i in range(point_count):
        line = pts_file.readline().split(" ")
        x_points.append(float(line[0]))
        y_points.append(float(line[1]))

    pts_file.close()
    return x_points, y_points



def show_flm_on_image_single(pts_file, image_file, block):

    # cv2.namedWindow("display")
    plt.figure()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_points, y_points = read_pts(pts_file)
    left, top, right, bottom = compute_pts_bb(x_points, y_points)


    # image = cv2.imread(image_file)
    image = mpimg.imread(image_file)
    # plt.imshow(image)
    #image = image[int(top):int(bottom), int(left):int(right)]
    # cv2.setWindowTitle("display", image_file)
    current_width = image.shape[1]
    new_width = current_width*1
    scale_factor = float(new_width) / current_width
    new_height = int(scale_factor * image.shape[0])

    t_x_points = [((x )* scale_factor) for x in x_points]
    t_y_points = [( (y) * scale_factor)  for y in y_points]
    image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
    i = 0
    for (x, y) in zip(t_x_points, t_y_points):
        #plt.plot()
        cv2.circle(image, (int(x) , int(y)), 2, (0, 0, 255), -1)
        # cv2.putText(image,"%d" % (i),(int(x) + random.randint(0, 10),int(y) + random.randint(0, 10)), font, 0.5,(255,255,255))
        i = i + 1
    # cv2.imshow('display',image)
    plt.imshow(image)
    plt.show(block=block)
    # plt.show()
    # draw()

    #    return
    # show()
    # cv2.destroyAllWindows()


def show_flm_on_image(directory):
    cv2.namedWindow("display")

    font = cv2.FONT_HERSHEY_SIMPLEX

    #cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    for filename in glob.glob(directory):
        if filename.endswith(".pts"):

            directory = os.path.dirname(filename)
            filename = os.path.basename(filename)
            x_points, y_points = read_pts(directory + "//" + filename)
            left, top, right, bottom = compute_pts_bb(x_points, y_points)


            image_file_name = filename[0:len(filename) - 3] + "jpg"
            image = cv2.imread(directory + "//" + image_file_name)
            image = image[int(top):int(bottom), int(left):int(right)]
            cv2.setWindowTitle("display", image_file_name)

            current_width = image.shape[1]
            new_width = 400
            scale_factor = float(new_width) / current_width
            new_height = int(scale_factor * image.shape[0])

            t_x_points = [((x - left )* scale_factor) for x in x_points]
            t_y_points = [( (y - top) * scale_factor)  for y in y_points]

            image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
            i = 0
            for (x, y) in zip(t_x_points, t_y_points):
                #plt.plot()
                # cv2.circle(image, (int(x) , int(y)), 2, (0, 0, 255), -1)
                if i%2:
                    image = cv2.putText(image,"%d" % (i),(int(x),int(y)), font, 0.2,(255,255,255))
                i = i + 1

            #image = cv2.resize(image, (1500, 1500), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('display',image)
            if cv2.waitKey(0) == ord('q'):
                break

    cv2.destroyAllWindows()




########################################################################################################################


def render_flm_on_top_of_image_in_dir(image_dir, flm_dir, bbs_dir, out_dir):
    for flm_name in glob.glob(flm_dir):
        flm_base = os.path.basename(flm_name)
        image = cv2.imread(image_dir + '/' + flm_base[:-3] + 'jpg')
        x_points, y_points = read_pts(flm_name)

        bbs_file = bbs_dir + '/' + flm_base

        # if not os.path.isfile(bbs_file):
        x0 = np.min(x_points)
        x1 = np.max(x_points)
    # else:
        # x0,y0,x1,y1= read_rect_from_file(bbs_file)
        flm_size = (x1 - x0) / 80.0
        for (x, y) in zip(x_points, y_points):
            cv2.circle(image, (int(x) , int(y)), int(flm_size), (0, 0, 255), -1)

        output_file = flm_name[:-3] + 'jpg'
        cv2.imwrite(output_file, image)


########################################################################################################################


def write_pts(pts_output_file, points):

    num_pts = points.shape[0]
    #print('Writing '+str(num_pts)+' pts')
    pts_file = open(pts_output_file,'w')
    pts_file.write("version: 1\n")
    pts_file.write("n_points: "+str(num_pts)+"\n")
    pts_file.write("{\n")
    for (x, y) in points:
        pts_file.write("%.3f %.3f\n" % (x, y) )

    pts_file.write("}\n")
    pts_file.close()


########################################################################################################################


def render_face(image_file, flm_file, bbs_file, output_file, render_id=False):
    image = cv2.imread(image_file)
    x_points, y_points = read_pts(flm_file)
    x_points = np.round(x_points)
    y_points = np.round(y_points)
    if os.path.isfile(bbs_file):
        x0,y0,x1,y1= read_rect_from_file(bbs_file)
        cv2.rectangle(image, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,255))

    x0 = np.min(x_points)
    x1 = np.max(x_points)
    flm_size = (x1 - x0) / 80.0
    try:
        if render_id:
            scale_factor = 3
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        for i,(x, y) in enumerate(zip(x_points, y_points)):
            if render_id:
                cv2.putText(image, str(i), (int(x * scale_factor), int(y * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
            else:
                cv2.circle(image, (int(x) , int(y)), int(flm_size), (0, 0, 255), -1)
    except:
        print 'oh no'

    #output_file = flm_file[:-3] + 'jpg'
    cv2.imwrite(output_file, image)
    print('written ' + output_file)


########################################################################################################################

def fix_pupils_for_flm(flm_dir):
    for root, dirnames, filenames in os.walk(flm_dir):
        for filename in fnmatch.filter(filenames, '*.pts'):
            full_path = os.path.join(root, filename)
            x_points, y_points = read_pts(full_path)
            x_points[163] = x_points[99]
            y_points[163] = y_points[99]
            x_points[164] = x_points[100]
            y_points[164] = y_points[100]

            write_pts(full_path, np.array(zip(x_points, y_points)))





########################################################################################################################

def render_flm_on_top_of_image_recursive(image_dir, flm_dir, bbs_dir, out_dir, render_id=False):

    for root, dirnames, filenames in os.walk(flm_dir):
        for filename in fnmatch.filter(filenames, '*.pts'):
            if not os.path.exists(root):
                os.makedirs(root)
            full_path = os.path.join(root, filename)
            diff_path = root[len(flm_dir):] + '/'
            bbs_file = bbs_dir + '/' +  diff_path + '/' + filename
            image_file = image_dir + diff_path + filename[:-3] + 'jpg'
            output_file = out_dir + diff_path + filename[:-3] + 'jpg'
            if not os.path.exists(out_dir + diff_path ):
                os.makedirs(out_dir + diff_path)
            if not os.path.exists(image_file):
                image_file = image_dir + diff_path + filename[:-3] + 'png'
            render_face(image_file, full_path, bbs_file, output_file, render_id )


########################################################################################################################

if __name__ == "__main__":
    image_dir = "/opt/kwtc/WFLW/test/all/images/"
    estimated_flm = '/opt/kwtc/output/' 
    bbs_dir = "/opt/kwtc/WFLW/bb/images/"
    out_dir = '/tmp/'

    # fix_pupils_for_flm(estimated_flm)
    render_flm_on_top_of_image_recursive(image_dir, estimated_flm, bbs_dir, out_dir, False)
