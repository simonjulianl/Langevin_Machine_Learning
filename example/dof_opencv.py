import matplotlib.pyplot as plt
import numpy as np
import cv2

def compute_dense_optical_flow(prev_image, current_image):
    old_shape = current_image.shape
    # prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    # current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    assert current_image.shape == old_shape
    stacked_img = np.concatenate((prev_image,) * 3, axis=-1)
    print(stacked_img.shape)
    stacked_img = stacked_img.astype(np.float32)
    hsv = np.zeros_like(stacked_img)
    hsv[..., 1] = 255
    flow = None

    # levels : number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    # iterations: number of iterations the algorithm does at each pyramid level.
    # winsize : averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection,
    # but yield more blurred motion field.
    # poly_n : size of the pixel neighborhood used to find polynomial expansion in each pixel
    # poly_sigma : standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
    flow = cv2.calcOpticalFlowFarneback(prev=prev_image,
                                        next=current_image, flow=flow,
                                        pyr_scale=0.85, levels=1, winsize=1,
                                        iterations=1, poly_n=7, poly_sigma=1.5,
                                        flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print(mag,ang)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    print(hsv)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # return flow


frame1 = np.asarray(
    [[0, 0, 0, 0, 0, 0],
     [0.25, 0.5, 0.25, 0, 0, 0],
     [0.5, 1, 0.5, 0, 0, 0],
     [0.25, 0.5, 0.25, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]])

frame2 = np.asarray(
    [[0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0.25, 0.5, 0.25, 0, 0],
     [0, 0.5, 1, 0.5, 0, 0],
     [0, 0.25, 0.5, 0.25, 0, 0],
     [0, 0, 0, 0, 0, 0]])

frame1 = np.expand_dims(frame1, axis=-1)  # channel 1
frame2 = np.expand_dims(frame2, axis=-1)  # channel 1
# plt.imshow(frame1,cmap='gray')
# plt.show()
# plt.imshow(frame2,cmap='gray')
# plt.show()

rgb = compute_dense_optical_flow(frame1, frame2)
cv2.imshow('frame2',rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

draw_flow

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(
        np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - \
                                  np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
    2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - \
                                  np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
    0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - \
                                  np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """
    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def flow2img2(flow_data):
    """
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def visualize_flow_file(flow_data):
    img = flow2img2(flow_data)
    imgplot = plt.imshow(img)
    plt.show()  # ; plt.colorbar()


### This is for both U and V vectors ###
# visualize_flow_file(flow_vectors)
#
# ### To separate U and V, we can zero either one ###
# u = flow_vectors.copy()
# u[:, :, 1] = 0  # zero the V component
# v = flow_vectors.copy()
# v[:, :, 0] = 0  # zero the U component
# visualize_flow_file(u)
# visualize_flow_file(v)
