import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class momentum_fields:

    def __init__(self, phi_field_in, phi_field_nx):

        self._phi_field_in = phi_field_in
        self._phi_field_nx = phi_field_nx


    def compute_dense_optical_flow(self, prev_image, current_image):

        # first 8-bit single-channel input image
        assert current_image.shape == prev_image.shape

        flow = None
        flow = cv2.calcOpticalFlowFarneback(prev=prev_image,
                                            next=current_image, flow=flow,
                                            pyr_scale=0.85, levels=1, winsize=3,
                                            iterations=1, poly_n=7, poly_sigma=1.5,
                                            flags=0)

        return flow

    def p_field(self):

        nsamples, channels, npixels, npixels = self._phi_field_in.shape
        flow_vectors = torch.zeros((nsamples, npixels, npixels, 2))  # nsamples x npixels

        for z in range(nsamples):

            phi_field_in = self._phi_field_in[z].permute(1,2,0) # channel first to channel last for optical flow
            phi_field_nx = self._phi_field_nx[z].permute(1,2,0)

            # convert torch to array : to use opencv
            numpy_phi_field_in = phi_field_in.numpy()
            numpy_phi_field_nx = phi_field_nx.numpy()

            flow_v = self.compute_dense_optical_flow(numpy_phi_field_in, numpy_phi_field_nx)
            flow_vectors[z] = torch.from_numpy(flow_v)

        flow_vectors = flow_vectors.permute(0,3,1,2) # channel last to channel first for cnn

        return flow_vectors

    def flow2img2(self, flow_data):
        """
        convert optical flow into color image
        :param flow_data:
        :return: color image
        """
        # print(flow_data.shape)
        # print(type(flow_data))
        vx = flow_data[0, 0, :, :] # take one sample that is index=0
        vy = flow_data[0, 1, :, :]

        # # to make channel
        # height, width = vx.shape
        # imgvx = torch.zeros((height, width, 1))
        # imgvy = torch.zeros((height, width, 1))

        minm = min(vx.min(), vy.min())
        maxm = max(vx.max(), vy.max())
        # print(vx.min(), vy.min())
        # print(vx.max(), vy.max())
        # print(minm,maxm)
        #normalize img to 0-255
        # to show img
        norm_vx = (vx - minm) * 255 / (maxm - minm + 1e-5)
        norm_vy = (vy - minm) * 255 / (maxm - minm + 1e-5)

        # for i in range(height):
        #     for j in range(width):
        #
        #         imgvx[i,j,:] = norm_vx[i,j]
        #         imgvy[i,j, :] = norm_vy[i,j]

        return norm_vx, norm_vy

    def visualize_flow_file(self,flow_data):
        imgvx, imgvy = self.flow2img2(flow_data)
        plt.title('vx')
        plt.imshow(imgvx, cmap='gray')
        plt.colorbar()
        plt.clim(0, 255)
        plt.show()

        plt.title('vy')
        plt.imshow(imgvy, cmap='gray')
        plt.colorbar()
        plt.clim(0, 255)
        plt.show()
        plt.close()