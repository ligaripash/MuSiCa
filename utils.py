import numpy as np
from menpo.shape import PointCloud
import cv2

jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)

mirrored_parts_68 = np.hstack([
    jaw_indices[::-1], rbrow_indices[::-1], lbrow_indices[::-1],
    upper_nose_indices, lower_nose_indices[::-1],
    np.roll(reye_indices[::-1], 4), np.roll(leye_indices[::-1], 4),
    np.roll(outer_mouth_indices[::-1], 7),
    np.roll(inner_mouth_indices[::-1], 5)
])


def render_points_on_image(image, point_set_a, point_set_b, point_set_c, mse=0):
    # render points on the first image of the batch
    out_image = np.array(image[0,:,:,:]*255, dtype=np.uint8)
    # print ('shape', str(out_image.shape))

    for point in point_set_a[0,:,:]:
        # print (int(point[1]), int(point[0]))
        try:
            out_image[int(point[0]), int(point[1])] = (255,0,0)
        except:
            continue

    for point in point_set_b[0,:,:]:
        try:
            out_image[int(point[0]), int(point[1])] = (0, 255, 0)
        except:
            continue
        # out_image = cv2.circle(out_image, (int(point[1]), int(point[0])), 1, (0,255,0))

    for point in point_set_c[0,:,:]:
        try:
            out_image[int(point[0]), int(point[1])] = (0, 0, 255)
        except:
            continue

    # cv2.imwrite('/tmp/im_' + str(out_image.shape[1]) + '.jpg', out_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = out_image.shape[1] / 100.0
    out_image = cv2.putText(out_image, str(mse), (0,out_image.shape[1]/2), font, font_scale,(255, 255, 255),1,cv2.LINE_AA)
    out_image = np.array(out_image, dtype=np.float32)
    new_out = np.reshape(out_image, (1,out_image.shape[0], out_image.shape[1], out_image.shape[2]))
    return new_out



def mirror_landmarks_68(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_68])


def mirror_image(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1].copy()

    for group in im.landmarks:
        lms = im.landmarks[group].lms
        if lms.points.shape[0] == 68:
            im.landmarks[group] = mirror_landmarks_68(lms, im.shape)

    return im


def mirror_image_bb(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1]
    im.landmarks['bounding_box'] = PointCloud(abs(np.array([0, im.shape[
        1]]) - im.landmarks['bounding_box'].lms.points))
    return im


def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color


def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img


def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])


def get_central_crop(images, box=(6, 6)):
    _, w, h, _ = images.get_shape().as_list()

    half_box = (box[0] / 2., box[1] / 2.)

    a = slice(int((w // 2) - half_box[0]), int((w // 2) + half_box[0]))
    b = slice(int((h // 2) - half_box[1]), int((h // 2) + half_box[1]))

    return images[:, a, b, :]


def build_sampling_grid(patch_shape):
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


default_sampling_grid = build_sampling_grid((30, 30))


def extract_patches(pixels, centres, sampling_grid=default_sampling_grid):
    """ Extracts patches from an image.

    Args:
        pixels: a numpy array of dimensions [width, height, channels]
        centres: a numpy array of dimensions [num_patches, 2]
        sampling_grid: (patch_width, patch_height, 2)

    Returns:
        a numpy array [num_patches, width, height, channels]
    """
    pixels = pixels.transpose(2, 0, 1)

    max_x = pixels.shape[-2] - 1
    max_y = pixels.shape[-1] - 1

    patch_grid = (sampling_grid[None, :, :, :] + centres[:, None, None, :]
                  ).astype('int32')

    X = patch_grid[:, :, :, 0].clip(0, max_x)
    Y = patch_grid[:, :, :, 1].clip(0, max_y)

    return pixels[:, X, Y].transpose(1, 2, 3, 0)
