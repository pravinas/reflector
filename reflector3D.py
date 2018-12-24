#! usr/bin/python

import numpy as np
import cv2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Surface3D():

    def __init__(self, function, image = None, bounds = None, mode="transparent"):
        self.function = function
        # "perfect", "lambertian", "transparent"
        self.mode = mode

        if image is None:
            if bounds is None:
                print("Image or bounds/step must be specified for Surface3D")
                raise Exception
            self.image = None
            self.ins = np.mgrid[bounds[0]:bounds[1]:bounds[2], bounds[0]:bounds[1]:bounds[2]]
        else:
            height, width, depth = image.shape
            self.image = image
            self.ins = np.mgrid[-height/2:height/2,-width/2:width/2]

        self.zs = self.function(self.ins[0], self.ins[1])
        self.gradients = np.gradient(self.zs)
        lenx = self.gradients[0].shape
        nx = np.stack((np.ones(lenx), np.zeros(lenx), self.gradients[0]),axis=0)
        leny = self.gradients[1].shape
        ny = np.stack((np.zeros(leny), np.ones(leny), self.gradients[1]),axis=0)
        self.normals = np.cross(nx, ny, axis = 0)/np.linalg.norm(np.cross(nx, ny, axis = 0),axis=0)
        #print(self.normals.shape)

    def height(self):
        return np.amax(self.zs)

    def reflect(self, input_rays):
        if self.mode == "perfect":
            #dim = self.normals.shape
            #i = np.identity(3)
            #i = i[None, None,...].repeat(dim[1], axis=0).repeat(dim[2], axis=1)
            #newnorm = np.zeros((dim[1],dim[2],3,3))
            #for ix0 in range(dim[1]):
            #    for iy0 in range(dim[2]):
            #        n = self.normals[:,ix0,iy0]
            #        newnorm[ix0,iy0] = i - 2*np.outer(n,n)
            #        #print(newnorm[ix0,iy0].shape)
            #return newnorm
            i = np.identity(3)[..., None, None]
            r_matrices = i - 2 * np.einsum("ixy,jxy->ijxy", self.normals, self.normals)
            reflections = np.einsum("ijxy,ixy->jxy",r_matrices,input_rays)
            return reflections

        if self.mode == "lambertian":
            pass
        if self.mode == "transparent":
            pass

    def plot_big_surface(self, axes):
        #imgcmap = cv2.applyColorMap(self.image, cv2.COLORMAP_JET)
        axes.contour3D(self.ins[0],self.ins[1],self.zs, 50)#, cmap=imgcmap)

    def plot_surface(self,axes):
        axes.plot_surface(self.ins[0],self.ins[1],self.zs)

    def plot_surface(self, axes, image):
        pass

    def plot_reflections(self, ax, reflections=None, input_rays=None):
        if reflections is None:
            if input_rays is None:
                input_rays = np.stack((np.zeros(self.ins.shape[1:]),np.zeros(self.ins.shape[1:]),-1*np.ones(self.ins.shape[1:])))
            reflections = self.reflect(input_rays)

        ax.quiver(self.ins[0],self.ins[1],self.zs,reflections[0],reflections[1],reflections[2], length=1, normalize=True)
        ax.quiver(self.ins[0],self.ins[1],self.zs,input_rays[0],input_rays[1],input_rays[2], length=1, normalize=True,color='orange')




class Reflector3D():
    def __init__(self, image, mirrorfn, collectorfn):
        '''
        image - an opencv image
        mirror - a mirror object
        collector - a collector object
        zimg - the height at which the image is above the xy plane
        '''
        # The image itself
        self.image = image

        # The reflecting surface of the image
        self.mirror = Surface3D(mirrorfn, image=image, mode="perfect")

        # The surface of this image which views the reflector
        # Will mostly be a straight line.
        # Will definitely be transparent.
        self.collector = Surface3D(collectorfn, image=image, bounds=(-5,5,0.01), mode="transparent")

        # The height above the plane at which the image sits
        self.zimg = self.set_height()

    def set_height(self, offset=0):
        self.zimg = self.mirror.height() + offset
        return self.zimg

    def display_plot(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.mirror.plot_big_surface(ax)
        self.mirror.plot_reflections(ax)

        #self.collector.plot_surface(ax)
        plt.show()

if __name__ == '__main__':
    image = cv2.imread('gradient_small.png')
    #mirror = lambda x,y: 0.001*x**2 + 0.001*y**2
    mirror = lambda x,y: x**2 + y**2
    collector = lambda x,y: 4+0*x+0*y
    r = Reflector3D(image, mirror, collector)
    r.display_plot()