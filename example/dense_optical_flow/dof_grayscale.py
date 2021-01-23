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
                                        pyr_scale=0.5, levels=3, winsize=3,
                                        iterations=10, poly_n=5, poly_sigma=1.1,
                                        flags=0)

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # print(mag,ang)
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow

mask = np.asarray(
    [[0.1, 0.1, 0.1, 0.1, 0.1, 0],
     [0.1,0.25, 0.5, 0.25, 0.1, 0],
     [0.1,0.5, 1, 0.5, 0.1, 0],
     [0.1,0.25, 0.5, 0.25, 0.1, 0],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0],
     [0, 0, 0, 0, 0, 0]])

frame1 = np.zeros((32,32))
frame2 = np.zeros((32,32))

x0=8
y0=8
x1=16
y1=8
frame1[x0:x0+6,y0:y0+6]= mask
frame2[x0+1:x0+1+6,y0:y0+6]= mask # move 1 pixel only x direction
frame1[x1:x1+6,y1:y1+6]= mask
frame2[x1+1:x1+1+6,y1+2:y1+2+6]= mask # move 1 pixel x direction and 2 pixel y direction
print(frame1[x0-1:,y0-1:])
print(frame2)
frame1 = np.expand_dims(frame1, axis=-1)  # channel 1
frame2 = np.expand_dims(frame2, axis=-1)  # channel 1
# plt.imshow(frame1,cmap='gray')
# plt.show()
# plt.imshow(frame2,cmap='gray')
# plt.show()
# quit()

flow_vectors = compute_dense_optical_flow(frame1, frame2)
# print(flow_vectors)
# print(flow_vectors.shape)

def flow2img2(flow_data):
    """
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
    # print(flow_data.shape)
    # print(type(flow_data))
    vx = flow_data[:, :, 0]
    vy = flow_data[:, :, 1]

    # print(vx,vy)

    height, width = vx.shape
    imgvx = np.zeros((height, width, 1))
    imgvy = np.zeros((height, width, 1))

    minm = min(vx.min(), vy.min())
    maxm = max(vx.max(), vy.max())
    #normalize img to 0-255

    # to show img
    norm_vx = (vx - minm) * 255 / (maxm - minm + 1e-5)
    norm_vy = (vy - minm) * 255 / (maxm - minm + 1e-5)
    # norm_u = np.abs((u - minu) * 255)
    # norm_v = np.abs((v - minu) * 255)


    # print(norm_u,norm_v)

    for i in range(height):
        for j in range(width):

            imgvx[i,j,:] = norm_vx[i,j]
            imgvy[i,j, :] = norm_vy[i,j]

    # # print('img',imgu,imgv)

    print('max ', maxm)
    print('min ', minm)
    return imgvx, imgvy


def visualize_flow_file(flow_data):
    imgvx, imgvy = flow2img2(flow_data)
    plt.title('vx')
    # print(imgu)
    plt.imshow(imgvx, cmap='gray')
    plt.colorbar()
    plt.clim(0, 255)
    plt.show()
    plt.title('vy')
    # print(imgv)
    plt.imshow(imgvy, cmap='gray')
    plt.colorbar()
    plt.clim(0, 255)
    plt.show()


### This is for both U and V vectors ###
visualize_flow_file(flow_vectors)

### To separate U and V, we can zero either one ###
# u = flow_vectors.copy()
# u[:, :, 1] = 0  # zero the V component
# v = flow_vectors.copy()
# v[:, :, 0] = 0  # zero the U component
# visualize_flow_file(u)
# visualize_flow_file(v)
