import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Img
from scipy import signal
from skimage.feature import corner_harris, corner_peaks
import cv2
import math


def compute_dense_optical_flow(prev_image, current_image):
    old_shape = current_image.shape
    # prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    # current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    assert current_image.shape == old_shape
    # hsv = np.zeros_like(prev_image)
    # hsv[..., 1] = 255
    flow = None

    # levels : number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    # iterations: number of iterations the algorithm does at each pyramid level.
    # winsize : averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection,
    # but yield more blurred motion field.
    # poly_n : size of the pixel neighborhood used to find polynomial expansion in each pixel
    # poly_sigma : standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
    flow = cv2.calcOpticalFlowFarneback(prev=prev_image,
                                        next=current_image, flow=flow,
                                        pyr_scale=0.5, levels=3, winsize=7,
                                        iterations=3, poly_n=5, poly_sigma=1.1,
                                        flags=0)

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # print(mag,ang)
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow

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

flow_vectors = compute_dense_optical_flow(frame1, frame2)
print('vx',flow_vectors[:,:,0])
print('vy',flow_vectors[:,:,1])

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

    height, width = u.shape
    imgu = np.zeros((height, width, 1))
    imgv = np.zeros((height, width, 1))

    minu = u.min()
    # print('mimu',minu)
    maxu = u.max()
    # print('maxu', maxu)
    minv = v.min()
    # print('minv', minv)
    maxv = v.max()
    # print('maxv', maxv)

    #normalize img to 0-255
    norm_u = (u - minu * 1) / (maxu - minu)
    norm_v = (v - minv * 1) / (maxv - minv)

    # print(norm_u,norm_v)

    for i in range(height):
        for j in range(width):

            imgu[j,i,:] = norm_u[j,i]
            imgv[j, i, :] = norm_v[j, i]

    print('img',imgu,imgv)

    return imgu, imgv


def visualize_flow_file(flow_data):
    imgu, imgv = flow2img2(flow_data)
    plt.title('vx')
    imguplot = plt.imshow(imgu, cmap='gray')
    plt.colorbar()
    plt.show()
    plt.title('vy')
    imgvplot = plt.imshow(imgv, cmap='gray')
    plt.colorbar()
    plt.show()  # ; plt.colorbar()


### This is for both U and V vectors ###
visualize_flow_file(flow_vectors)

### To separate U and V, we can zero either one ###
# u = flow_vectors.copy()
# u[:, :, 1] = 0  # zero the V component
# v = flow_vectors.copy()
# v[:, :, 0] = 0  # zero the U component
# visualize_flow_file(u)
# visualize_flow_file(v)
