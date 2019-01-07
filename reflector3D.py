#! usr/bin/python

import numpy as np
import cv2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Surface3D():

    def __init__(self, function, image=None, shape=None, bounds = None, mode="transparent"):
        '''
        function - 2-input function corresponding to this surface in 3d
        image - an image to project onto the surface
        bounds - (min, max)
        '''
        self.function = function
        # "perfect", "lambertian", "transparent"
        self.mode = mode

        if shape is not None:
            height, width, depth = shape
            self.image = np.zeros(shape)
            self.shape = shape

        if image is not None:
            height, width, depth = image.shape
            self.image = image
            self.shape = self.image.shape

        if bounds is None:
            self.points = np.mgrid[0:height,0:width]
        else:
            self.points = np.mgrid[bounds[0]:bounds[1]:height*1j, bounds[0]:bounds[1]:width*1j]


        zs = self.function(self.points[0], self.points[1])
        self.points = np.stack((self.points[0],self.points[1], zs))
        gradients = np.gradient(zs)
        lenx = gradients[0].shape
        nx = np.stack((np.ones(lenx), np.zeros(lenx), gradients[0]),axis=0)
        leny = gradients[1].shape
        ny = np.stack((np.zeros(leny), np.ones(leny), gradients[1]),axis=0)
        self.normals = np.cross(nx, ny, axis = 0)/np.linalg.norm(np.cross(nx, ny, axis = 0),axis=0)

        #make shapes more convenient
        self.points = self.points.reshape((3,-1))
        self.normals = self.normals.reshape((3,-1))


    def height(self):
        return np.amax(self.points[2])

    def reflect(self, input_rays):
        if self.mode == "perfect":
            i = np.identity(3)[..., None, None]
            height, width, rgb = self.shape
            # the next few lines are a hack b/c I don't understand
            # einstein summation.
            # TODO: get emily to help me fix this
            normals = self.normals.reshape((3,width,height))
            input_rays = input_rays.reshape((3, width, height))
            r_matrices = i - 2 * np.einsum("ixy,jxy->ijxy", normals, normals)
            reflections = np.einsum("ijxy,ixy->jxy",r_matrices,input_rays)
            return reflections.reshape((3,-1))

        if self.mode == "lambertian":
            pass
        if self.mode == "transparent":
            return input_rays

    def plot_surface(self, axes):
        axes.contour3D(self.points[0],self.points[1],self.points[2], 50)

    def plot_surface_img(self, axes, alpha=0.8):
        dim, n_points = self.normals.shape
        height, width, rgb = self.shape
        for i in range(n_points):
            point = self.points.T[i]
            normal = self.normals.T[i]
            # a plane is a*x+b*y+c*z+d=0
            # [a,b,c] is the normal. Thus, we have to calculate
            # d and we're set
            d = -point.dot(normal)

            # create x,y
            xx, yy = np.mgrid[-0.5:0.5:3j, -0.5:0.5:3j]
            xx = xx + point[0]
            yy = yy + point[1]

            # calculate corresponding z
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

            # plot the surface
            axes.plot_surface(xx, yy, z, color=list(self.image[int(i/width),i%width,:]/255.)+[alpha])


    def plot_reflections(self, ax, reflections=None, input_rays=None):
        if input_rays is None:
            # Default rays straight down
            ir_shape = self.points.shape[1:]
            input_rays = np.stack((np.zeros(ir_shape),np.zeros(ir_shape),-1*np.ones(ir_shape)))

        if reflections is None:
            reflections = self.reflect(input_rays)

        #ax.quiver(self.points[0],self.points[1],self.points[2],reflections[0],reflections[1],reflections[2], length=1, normalize=True)
        #ax.quiver(self.points[0],self.points[1],self.points[2],input_rays[0],input_rays[1],input_rays[2], length=1, normalize=True,color='orange')

    def collect(self, mirror, input_rays=None, tolerance=0.7):
        # if mirror.focus is not None:
        #     # draw a line from each point on the collector to the focus
        #     dim, width, height = self.points.shape
        #     mdim, mwidth, mheight = mirror.points.shape
        #     newimg = np.zeros(self.shape)
        #     for ix in range(width):
        #         for iy in range(height):
        #             f = mirror.focus
        #             a = 2 * np.sqrt(f) * np.sign(f)
        #             x,y,z = self.points[:,ix,iy]
        #             kp = (z-f + np.sqrt((z-f)**2 + 4*f*(a*x*x + a*y*y)))/(2*(a*x*x + a*y*y))
        #             km = (z-f - np.sqrt((z-f)**2 + 4*f*(a*x*x + a*y*y)))/(2*(a*x*x + a*y*y))
        #             ppoint = np.array([kp*x, kp*y, kp*z-kp*f+f]).T
        #             mpoint = np.array([km*x, km*y, km*z-km*f+f]).T

        #             # mpoint and ppoint are the two points on the paraboloid which
        #             # reflect into x,y,z. Now to get their colors and add them up.
        #             # just going to use ppoint for now

        #             xin = int(np.around(ppoint[0] + mwidth/2 - 0.01))
        #             yin = int(np.around(ppoint[1] + mheight/2 - 0.01))
        #             newimg[xin,yin] = mirror.image[xin, yin]
        #     self.image = newimg
        # else:
        if input_rays is None:
            # Default rays straight down
            ir_shape = mirror.points.shape[1:]
            input_rays = np.stack((np.zeros(ir_shape),np.zeros(ir_shape),-1*np.ones(ir_shape)))
        reflection_rays = mirror.reflect(input_rays)

        # for each ray, calculate points on the collection surface for which
        # the ray is within a small tolerance of.
        # That is, check if a point P is on-ish the line given by a point P_l and ray r_l
        newimg = np.zeros(self.shape)
        nheight, nwidth, ndepth = self.shape

        dimm, m_points = mirror.points.shape
        mheight, mwidth, mdepth = mirror.shape

        dimc, c_points = self.points.shape

        for i in range(m_points):
            m_point = mirror.points.T[i]
            ray = reflection_rays.T[i]
            pixel = mirror.image[int(i/mwidth),i%mwidth]
            for j in range(c_points):
                c_point = self.points.T[j]
                norm = np.linalg.norm(m_point-c_point)
                cos = np.dot(ray, (m_point-c_point))/norm
                distance = norm * np.sqrt(1-cos**2)
                if distance < tolerance:
                    newimg[int(j/nwidth),j%nwidth] = pixel

        self.image = newimg.astype(np.uint8)







class Reflector3D():
    def __init__(self, image, mirror, collectorfn):
        '''
        image - an opencv image
        mirror - a mirror object
        collector - a collector object
        zimg - the height at which the image is above the xy plane
        '''
        # The image itself
        self.image = image

        # The reflecting surface of the image
        #self.mirror = Surface3D(mirrorfn, image=image, focus=0.25, mode="perfect")
        self.mirror = mirror 

        # The surface of this image which views the reflector
        # Will mostly be a straight line.
        # Will definitely be transparent.
        self.collector = collector 
        self.collector.collect(self.mirror)

        # The height above the plane at which the image sits
        self.zimg = self.set_height()


    def set_height(self, offset=0):
        self.zimg = self.mirror.height() + offset
        return self.zimg

    def show_collected_img(self):
        cv2.imshow("input img", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        cv2.imshow("collected data", cv2.cvtColor(self.collector.image, cv2.COLOR_RGB2BGR))

    def display_plot(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #plt.imshow(self.image)
        self.mirror.plot_surface_img(ax, alpha=0.2)
        self.mirror.plot_reflections(ax)

        self.collector.plot_surface_img(ax, alpha=1)
        plt.show()

if __name__ == '__main__':
    image = cv2.imread('symmetric.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w,d = image.shape

    #mirrorfn = lambda x,y: (x-h/2.)**2 + (y-w/2.)**2
    mirrorfn = lambda x,y: np.sin(0.03*((x-h/2.)+(y-w/2)**2))
    mirror = Surface3D(mirrorfn, image=image, mode="perfect")

    collectorfn = lambda x,y: 0.5+1*np.sin(0.4*x)-np.sin(0.3*y**2)
    collector = Surface3D(collectorfn, shape=(40,40,3), bounds=(min(h,w)/2 - 10, max(h,w)/2+10), mode="transparent")

    r = Reflector3D(image, mirror, collector)
    r.show_collected_img()
    r.display_plot()