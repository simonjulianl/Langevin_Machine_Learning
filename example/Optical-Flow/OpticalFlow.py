import math
from scipy import signal
from PIL import Image
import numpy
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
import random

def lkwop (inp1, inp2, t ):

    H = array(Image.open(inp1).convert('L')) # read the first input frame
    I = array(Image.open(inp2).convert('L')) # read the second input frame

    # First Derivative in X direction
    fx = signal.convolve2d(H,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(I,[[-0.25,0.25],[-0.25,0.25]],'same')
    # First Derivative in Y direction
    fy = signal.convolve2d(H,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(I,[[-0.25,-0.25],[0.25,0.25]],'same')
    # First Derivative in XY direction
    ft = signal.convolve2d(H,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(I,[[-0.25,-0.25],[-0.25,-0.25]],'same')

    # Determining the good features
    gf = cv2.goodFeaturesToTrack(H # Input image
    ,10000 # max corners
    ,0.01 # lambda 1 (quality)
    ,10 # lambda 2 (quality)
    )

    # Initializing the u and v arrays with non numbers so that we don't have to plot the irrelevant vectors later
    u = numpy.nan * numpy.ones(shape=(len(fx[:,0]),len(fx[0,:])))
    v = numpy.nan * numpy.ones(shape=(len(fx[:,0]),len(fx[0,:])))

    # Calculating the u and v arrays for the good features obtained n the previous step.
    for a in gf:
            j,i = a.ravel()
            # calculating the derivatives for the neighbouring pixels
            # since we are using  a 3*3 window, we have 9 elements for each derivative.
            i = int(i)
            j = int(j)
            FX = ([fx[i-1,j-1],fx[i,j-1],fx[i-1,j-1],fx[i-1,j],fx[i,j],fx[i+1,j],fx[i-1,j+1],fx[i,j+1],fx[i+1,j-1]]) #The x-component of the gradient vector
            FY = ([fy[i-1,j-1],fy[i,j-1],fy[i-1,j-1],fy[i-1,j],fy[i,j],fy[i+1,j],fy[i-1,j+1],fy[i,j+1],fy[i+1,j-1]]) #The Y-component of the gradient vector
            FT = ([ft[i-1,j-1],ft[i,j-1],ft[i-1,j-1],ft[i-1,j],ft[i,j],ft[i+1,j],ft[i-1,j+1],ft[i,j+1],ft[i+1,j-1]]) #The XY-component of the gradient vector

            # Using the minimum least squares solution approach
            A = (FX,FY)
            A = matrix(A)
            AT = array(matrix(A)) # transpose of A
            A = array(numpy.matrix.transpose(A))

            U1 = numpy.dot(AT,A) #Psedudo Inverse
            U2 = numpy.linalg.pinv(U1)
            U3 = numpy.dot(U2,AT)
            (u[i,j],v[i,j]) = numpy.dot(U3,FT) # we have the vectors with minimized square error

    #======= Pick Random color for vector plot========
    colors = "bgrcmykw"
    color_index = random.randrange(0,8)
    c=colors[color_index]
    #======= Plotting the vectors on the image========
    plt.figure()
    plt.imshow(H,cmap = cm.gray)
    plt.title('Vector Plot of Good features on top of the Image')
    for m in range(1,len(fx[:-1,0])):
        for n in range(1,len(fx[0,:-1])):
            if abs(u[m,n])>t or abs(v[m,n])>t: # setting the threshold to plot the vectors
                plt.arrow(n,m,v[m,n],u[m,n],head_width = 5, head_length = 5, color = c)

    show()

t = 0.3 # choose threshold value
# lkwop('nparticle2_f1_pos.png','nparticle2_f2_pos.png',t)
# lkwop('basketball1.png','basketball2.png',t)
# lkwop('grove1.png','grove2.png',t)
lkwop('teddy1.png','teddy2.png',t)