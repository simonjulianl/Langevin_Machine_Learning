#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:01:47 2020

@author: simon
"""
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt

def project(N,DIM,BoxSize, frame ,xlist,ylist,zlist,scale = False):
    if (DIM == 2):
        
        figure = plt.figure(figsize = (10,10))
        ax = plt.axes(xlim = (-0.5 * BoxSize,0.5 * BoxSize) ,ylim = (-0.5*BoxSize,0.5*BoxSize))
        
        lines = []
        for i in range(N):
            lobj = ax.plot([],[],'o',markersize = 20)[0]
            lines.append(lobj)
            
        plt.xlabel('x/sigma')
        plt.ylabel('y/sigma')
        
        #initialize to empty list
        def init():
            for line in lines:
                line.set_data([],[])
            return lines
        
        data = []
        for i in range(N):
            data.append([xlist[i],ylist[i]])
       
            
        data = np.array(data)
        
        def animate(i):
            for line,datum in zip(lines,data):
                if scale :
                    line.set_data(datum[:,i-1:i] * BoxSize)
                else:
                    line.set_data(datum[:,i-1:i])
            return lines
        
        anim = animation.FuncAnimation(figure,animate,frames = frame, init_func = init,interval =100 ,blit = True,repeat = False)
        anim.save('particle.gif',writer ='imagemagick',fps = 10)
    
    if (DIM == 3):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.cla()
        data = []
        for i in range(N):
            data.append([xlist[i],ylist[i],zlist[i]])
            
        data = np.array(data)
        
        #apparently in 3d plot, you cannot set into empty array
        lines = [ax.plot(dat[0,0:1],dat[1,0:1],dat[2,0:1],'o',markersize = 15)[0] for dat in data]
        
        def update_lines(num):
            for line,datum in zip(lines,data):
                if scale :
                    line.set_data(datum[0:2,num-1:num] * BoxSize) #rescale to the BoxSize
                    line.set_3d_properties(datum[2,num-1:num] * BoxSize) #for 2D line, 3D animation should be set for z data 
                else:
                    line.set_data(datum[0:2,num-1:num])
                    line.set_3d_properties(datum[2,num-1:num])
                line.set_marker('o')
            return lines
        
        ax.set_xlim(-0.5 * BoxSize,0.5 * BoxSize)
        ax.set_ylim(-0.5 * BoxSize,0.5 * BoxSize)
        ax.set_zlim(-0.5 * BoxSize,0.5 * BoxSize)
        
        ax.set_ylabel('y/sigma')
        ax.set_xlabel('x/sigma')
        ax.set_zlabel('z/sigma')
        
        #save the animation
        line_ani = animation.FuncAnimation(fig,update_lines,frames = frame, fargs = None,interval = 100,blit = True , repeat = False)
        line_ani.save('particle_3D.gif', writer='imagemagick',fps=10)
        

    