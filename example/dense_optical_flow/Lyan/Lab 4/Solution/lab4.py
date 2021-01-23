""" CS4243 Lab 4: Tracking
Please read accompanying Jupyter notebook (lab4.ipynb) and PDF (lab4.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time


# Part 1 

def meanShift(dst, track_window, max_iter=100,stop_thresh=1):
    """Use mean shift algorithm to find an object on a back projection image.

    Args:
        dst (np.ndarray)            : Back projection of the object histogram of shape (H, W).
        track_window (tuple)        : Initial search window. (x,y,w,h)
        max_iter (int)              : Max iteration for mean shift.
        stop_thresh(float)          : Threshold for convergence.
    
    Returns:
        track_window (tuple)        : Final tracking result. (x,y,w,h)


    """

    completed_iterations = 0
    
    """ YOUR CODE STARTS HERE """
    # Search for each pixel within the tracking window, take a window of same size around it and compute the mean. 
    # The pixel that has the largest mean is the new center of tracking window. 
    # Iteratively update and deal with cases where we cannot take windows for pixels (that are at the edge of the img)

    # dst is basically a matrix of the same size as the image but each pixel represents the probability that it belongs to the feature we want to track
    # If we center the window right on the feature we want (e.g. hand), we will have the highest mean over the values in the window and that is the new
    # center for target in this frame and we return that 
    H,W = dst.shape

    # The x and y are at the top left corner of the box
    x,y,w,h = track_window

    # For convenience so we don't have to keep recalculating to take x - w//2 to x + w//2 etc
    offset_w, offset_h = w//2, h//2

    # These are the top left corner coordinates
    curr_x, curr_y = x, y

    # Keeps track of the previous centers
    prev_x = curr_x + offset_w 
    prev_y = curr_y + offset_h

    n_iter = 0
    while (1):
        #print(str(curr_x) + " " + str(curr_y))
        sum_of_weights = 0
        weighted_x = 0
        weighted_y = 0

        # If the entire window is within the range, do this faster method
        if curr_x >= 0 and curr_y >= 0 and curr_x + w < W and curr_y + h < H:
            # Create the x indices and y indices as a list and make use of python broadcasting to vectorize
            x_indices = np.array([val for val in range(curr_x,curr_x + w)]).reshape(1,-1)
            y_indices = np.array([val for val in range(curr_y, curr_y + h)]).reshape(-1,1)

            window_of_weights = dst[curr_y:curr_y+h, curr_x:curr_x+w]
            sum_of_weights = np.sum(window_of_weights)
            # Gets the weighted sum of the window. e.g. x_indices is broadcast to the same shape as window_of_weights and elementwise multiplication is done
            weighted_x = np.sum(window_of_weights * x_indices)
            weighted_y = np.sum(window_of_weights * y_indices)

        # Else iteratively do the slower method
        else:
            start_x = -1
            start_y = -1
            sub_x_len = 0
            sub_y_len = 0
            # Need to find out which side of the bounding box is out of the frame
            # Bounding box on left of frame
            if curr_x < 0:
                # Sub-Window will start at 0
                start_x = 0
                # Length of the subwindow in x-dir
                sub_x_len = w + curr_x
            # If bounding box on right of frame
            elif curr_x + w >= W:
                start_x = curr_x
                sub_x_len = W - start_x
            # Else, entire x portion of window is still within frame and start_x is just the top left corner x value (curr_x) and sub_x_len is w
            else:
                start_x = curr_x
                sub_x_len = w
            # If the bounding box is above the top of frame
            if curr_y < 0:
                # Subwindow of y starts at index 0
                start_y = 0
                # Length of subwindow in y-dir
                sub_y_len = h + curr_y
            # If the bounding box has a part below the bottom of frame
            elif curr_y + h >= H:
                # Subwindow starts at same top left corner
                start_y = curr_y
                # Length of subwindow in y-dir
                sub_y_len = H - start_y
            # Else, the bounding box along y-axis is entirely within the image. Just set start_y to top left corner y value (curr_y) and len to h
            else:
                start_y = curr_y
                sub_y_len = h

            end_x = start_x + sub_x_len
            end_y = start_y + sub_y_len
            # Define the x and y index lists that we will use for broadcasting
            x_indices = np.array([val for val in range(start_x, end_x)]).reshape(1,-1)
            y_indices = np.array([val for val in range(start_y, end_y)]).reshape(-1,1)

            # Get the slice in dst corresponding to the subwindow
            window_of_weights = dst[start_y:end_y,start_x:end_x]
            sum_of_weights = np.sum(window_of_weights)
            # Get the weighted sum of the window
            weighted_x = np.sum(window_of_weights * x_indices)
            weighted_y = np.sum(window_of_weights * y_indices)

            """
            # Loop through the window
            for i in range(h):
                for j in range(w):
                    if curr_x+j < 0 or curr_x+j >= W or curr_y+i < 0 or curr_y+i >= H:
                        continue 

                    
                    # Calculate the weighted value of each x-coord/y-coord with the weight of that cell
                    weighted_x += dst[curr_y+i,curr_x+j] * (curr_x+j)
                    weighted_y += dst[curr_y+i, curr_x+j] * (curr_y+i)
                    # Calculate the sum of the weights
                    sum_of_weights += dst[curr_y+i, curr_x+j]
            """
        # Get the new centroid coordinates. Use rounding to get closest (x,y)
        center_x = int(np.round(weighted_x/sum_of_weights))
        center_y = int(np.round(weighted_y/sum_of_weights))

        # Set the new top left corner
        curr_x = center_x - offset_w
        curr_y = center_y - offset_h

        """

        weight_slice = dst[y:y + h + 1, x:x + w + 1]
        weighted_coords_x = 0
        weighted_coords_y = 0
        weighted_sum = 0

        for i in range(h):
            for j in range(w):
                weighted_coords_x += j*weight_slice[i,j]
                weighted_coords_y += i*weight_slice[i,j]

                weighted_sum += weight_slice[i,j]

        shift_x = weighted_coords_x/weighted_sum - (x + offset_w)
        shift_y = weighted_coords_y/weighted_sum - (y + offset_h)

        print(shift_x)
        print(shift_y)
        
        # Take care of corner cases when window goes out of frame
        if prev_x + shift_x < 0 or prev_x + shift_x > W-1 or prev_y + shift_y < 0 or prev_y + shift_y > H-1:
            break
        # Shift top left coord of window (same as shifting that same amount from mid of window)
        new_x = prev_x + shift_x
        new_y = prev_y + shift_y
        """



        # Update num of iters
        n_iter += 1
        # Terminate if max_iters reached or change in locations is less than the threshold
        # Compare new center to previous center
        if n_iter == max_iter or np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2) < stop_thresh:
            break

        # Set the prev center to the current center to compare in the next iteration
        prev_x = center_x
        prev_y = center_x
    #print(n_iter)
    track_window = (curr_x, curr_y, w, h)

    """ YOUR CODE ENDS HERE """
    
    return track_window
    
    
    
        

def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes.

    Args:
        bbox1 (tuple)               : First bounding box position (x, y, w, h) where (x, y) is the top left corner of
                                      the bounding box, and (w, h) are width and height of the box.
        bbox2 (tuple)               : Second bounding box position (x, y, w, h) where (x, y) is the top left corner of
                                      the bounding box, and (w, h) are width and height of the box.
    Returns:
        score (float)               : computed IoU score.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0

    """ YOUR CODE STARTS HERE """

    
    # Get the bottom right of both the boxes
    x1b, y1b = x1 + w1, y1 + h1
    x2b, y2b = x2 + w2, y2 + h2
    # Compare the top left corners and bottom right corners to get the intersection box
    # Always take the higher x and higher y when comparing top left 
    # And the lower x and lower y when comparing bottom right
    x_it = max(x1, x2)
    y_it = max(y1, y2)
    x_ib = min(x1b, x2b)
    y_ib = min(y1b, y2b)

    # Compute the width and height of the intersection box
    i_width = x_ib - x_it
    i_height = y_ib - y_it

    # The area must be 0 if the top left's either coordinates are < bottom right's corresponding coordinate
    if i_width < 0 or i_height < 0:
        intersection = 0
    else:
        intersection = i_width * i_height

    # Unions is the sum of both areas - the intersection (if there was overlap there would be a double count of intersection)
    union = (x1b - x1) * (y1b - y1) + (x2b - x2) * (y2b - y2) - intersection

    score = intersection/union

    

    """ YOUR CODE ENDS HERE """

    return score


# Part 2:
def lucas_kanade(img1, img2, keypoints, window_size=9):
    """ Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.
    Returns:
        flow_vectors (np.ndarray)   : Estimated flow vectors for keypoints. flow_vectors[i] is
                                      the flow vector for keypoint[i]. Array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix.
    """
    assert window_size % 2 == 1, "window_size must be an odd number"
    
    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be loacated between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        """ YOUR CODE STARTS HERE """
        # Get the patch around the keypoint
        A_Ix = Ix[y-w:y+w+1, x-w:x+w+1].reshape(-1,1)
        A_Iy = Iy[y-w:y+w+1, x-w:x+w+1].reshape(-1,1)
        A = np.hstack((A_Ix, A_Iy))

        b = It[y-w:y+w+1, x-w:x+w+1].reshape(-1,1)
    
        # Transposing A
        A_T = np.transpose(A)
        # A^T . A
        AT_A = np.matmul(A_T,A)

        # Compute the x = [u v]^T as (A^T.A)^-1 . (A^T.b)
        uv_hat = np.matmul(np.linalg.inv(AT_A), np.matmul(A_T,b))

        flow_vectors.append([uv_hat[0,0], uv_hat[1,0]])
        """ YOUR CODE ENDS HERE """

    flow_vectors = np.array(flow_vectors)

    return flow_vectors


def compute_error(patch1, patch2):
    """ Compute MSE between patch1 and patch2
        - Normalize patch1 and patch2
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 (np.ndarray)         : Grayscale image patch1 of shape (patch_size, patch_size)
        patch2 (np.ndarray)         : Grayscale image patch2 of shape (patch_size, patch_size)
    Returns:
        error (float)               : Number representing mismatch between patch1 and patch2.
    """
    assert patch1.shape == patch2.shape, 'Differnt patch shapes'
    error = 0

    """ YOUR CODE STARTS HERE """
    patch_1_norm = (patch1 - np.mean(patch1))/np.std(patch1)
    patch_2_norm = (patch2 - np.mean(patch2))/np.std(patch2)

    #error = (1/(patch1.shape[0]*patch1.shape[1]))*np.sum((patch_1_norm - patch_2_norm)**2)
    error = np.mean((patch_1_norm - patch_2_norm)**2)

    """ YOUR CODE ENDS HERE """

    return error



def iterative_lucas_kanade(img1, img2, keypoints,
                           window_size=9,
                           num_iters=5,
                           g=None):
    """ Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.

        num_iters (int)             : Number of iterations to update flow vector.
        g (np.ndarray)              : Flow vector guessed from previous pyramid level.
                                      Array of shape (N, 2).
    Returns:
        flow_vectors (np.ndarray)   : Estimated flow vectors for keypoints. flow_vectors[i] is
                                      the flow vector for keypoint[i]. Array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2
    
   
    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)

    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2) # Initialize flow vector as zero vector
        y1 = int(round(y)); x1 = int(round(x))
        
        """ YOUR CODE STARTS HERE """

        # Get the patches of gradients to perform the computations in the patch around the keypoint
        Ix_patch, Iy_patch = Ix[y1 - w:y1 + w + 1, x1 - w: x1 + w + 1], Iy[y1 - w: y1 + w + 1, x1 - w: x1 + w + 1]
        # Compute G in the patch around that keypoint
        Ix2, Iy2, IxIy = np.sum(Ix_patch**2), np.sum(Iy_patch**2), np.sum(Ix_patch*Iy_patch)
        G = [[Ix2, IxIy], [IxIy, Iy2]]

        # For each iteration specified
        for k in range(num_iters):
            # Compute the temporal difference. gy and gx are basically the approximations of the actual estimated flow vector values from 
            # one level higher in the pyramid. Because each current estimation from naive LK only provides a small estimate that does not track large motions well
            # Calculate for only the relevant patch as thats the only thing we need for subsequent calculations
            # Have to use v[1] for the first axis and v[0] for the second axis because in line 406 vy is defined for index 0 and vx is defined for index 1
            # And the flow vector is (vy, vx) 
            del_Ik_patch = img1[y1 - w: y1 + w + 1,x1 - w: x1 + w + 1] - img2[y1 - w + int(gy) + int(v[1]): y1 + w + int(gy) + int(v[1]) + 1, x1 - w + int(gx) + int(v[0]): x1 + w + int(gx) + int(v[0]) + 1]
            # Compute image mismatch vector
            bk = [ np.sum(del_Ik_patch*Ix_patch), np.sum(del_Ik_patch*Iy_patch)]

            # Compute optical flow for that keypoint
            vk = np.matmul(np.linalg.inv(G), bk)
            v = v + vk
    

        """ YOUR CODE ENDS HERE """

        vx, vy = v
        flow_vectors.append([vy, vx])
        
    return np.array(flow_vectors)
        

def pyramid_lucas_kanade(img1, img2, keypoints,
                         window_size=9, num_iters=5,
                         level=2, scale=2):

    """ Pyramidal Lucas Kanade method

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.

        num_iters (int)             : Number of iterations to run iterative LK method
        level (int)                 : Max level in image pyramid. Original image is at level 0 of
                                      the pyramid.
        scale (float)               : Scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    """ YOUR CODE STARTS HERE """
    for L in reversed(range(len(pyramid1))):
        
        # Get the keypoint locations on this level of the pyramid
        new_keypoint_locs = keypoints/(scale**L)

        # Generate the optical flow using iterative LK at this level
        d = iterative_lucas_kanade(img1=pyramid1[L], img2=pyramid2[L], keypoints=new_keypoint_locs, g=g)

        # If it is not at level 0 (original image)
        if L != 0:
            # Get the guess g of the optical flow for the next level which is scale * (prev guess + optical flow from iterative LK)
            g = scale*(g + d)
    

    """ YOUR CODE ENDS HERE """

    d = g + d
    return d























"""Helper functions: You should not have to touch the following functions.
"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from skimage import filters, img_as_float
from skimage.io import imread
from skimage.transform import pyramid_gaussian

def load_frames_rgb(imgs_dir):

    frames = [cv2.cvtColor(cv2.imread(os.path.join(imgs_dir, frame)), cv2.COLOR_BGR2RGB) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_frames_as_float_gray(imgs_dir):
    frames = [img_as_float(imread(os.path.join(imgs_dir, frame), 
                                               as_gray=True)) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_bboxes(gt_path):
    bboxes = []
    with open(gt_path) as f:
        for line in f:
          
            x, y, w, h = line.split(',')
            #x, y, w, h = line.split()
            bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes

def animated_frames(frames, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_array(frames[i])
        return [im,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def animated_bbox(frames, bboxes, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    x, y, w, h = bboxes[0]
    bbox = ax.add_patch(Rectangle((x,y),w,h, linewidth=3,
                                  edgecolor='r', facecolor='none'))

    def animate(i):
        im.set_array(frames[i])
        bbox.set_bounds(*bboxes[i])
        return [im, bbox,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def animated_scatter(frames, trajs, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    scat = ax.scatter(trajs[0][:,1], trajs[0][:,0],
                      facecolors='none', edgecolors='r')

    def animate(i):
        im.set_array(frames[i])
        if len(trajs[i]) > 0:
            scat.set_offsets(trajs[i][:,[1,0]])
        else: # If no trajs to draw
            scat.set_offsets([]) # clear the scatter plot

        return [im, scat,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def track_features(frames, keypoints,
                   error_thresh=1.5,
                   optflow_fn=pyramid_lucas_kanade,
                   exclude_border=5,
                   **kwargs):

    """ Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3 # Take 3x3 patches to compute error
    w = patch_size // 2 # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i+1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi)); xi = int(round(xi))
            yj = int(round(yj)); xj = int(round(xj))
            # Point falls outside the image
            if yj > J.shape[0]-exclude_border-1 or yj < exclude_border or\
               xj > J.shape[1]-exclude_border-1 or xj < exclude_border:
                continue

            # Compute error between patches in image I and J
            patchI = I[yi-w:yi+w+1, xi-w:xi+w+1]
            patchJ = J[yj-w:yj+w+1, xj-w:xj+w+1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs