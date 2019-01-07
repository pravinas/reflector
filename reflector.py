import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt
import cv2

class Surface2D():
    def __init__(self, function, image, bounds = None, focus=None, mode="transparent"):
        '''
        function - function corresponding to this surface in 3d
        image - an image to project onto the surface
        bounds - (min, max)
        '''
        self.function = function
        # "perfect", "lambertian", "transparent"
        self.mode = mode
        self.focus = focus

        height, width, depth = image.shape
        self.image = image

        if bounds is None:
            self.points = np.mgrid[-width/2:width/2]
        else:
            self.points = np.mgrid[bounds[0]:bounds[1]:width*1j]


        self.derivatives = derivative(self.function, self.points)
        self.points = np.stack((self.points, self.function(self.points)))

    def height(self):
        return np.amax(self.points[2])

    def reflect(self, input_rays):
        if self.mode == "perfect":
            i = np.identity(3)[..., None, None]
            r_matrices = i - 2 * np.einsum("ixy,jxy->ijxy", self.normals, self.normals)
            reflections = np.einsum("ijxy,ixy->jxy",r_matrices,input_rays)
            return reflections

        if self.mode == "lambertian":
            pass
        if self.mode == "transparent":
            return input_rays

    def plot_surface(self, axes):
        axes.contour3D(self.points[0],self.points[1],self.points[2], 50)

    def plot_surface_img(self, axes, alpha=0.8):
        dim, width, height = self.normals.shape
        for ix in range(width):
            for iy in range(height):
                point = self.points[:,ix,iy]
                normal = self.normals[:,ix,iy]
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
                axes.plot_surface(xx, yy, z, color=list(self.image[ix,iy,:]/255.)+[alpha])


    def plot_reflections(self, ax, reflections=None, input_rays=None):
        if input_rays is None:
            # Default rays straight down
            ir_shape = self.points.shape[1:]
            input_rays = np.stack((np.zeros(ir_shape),np.zeros(ir_shape),-1*np.ones(ir_shape)))

        if reflections is None:
            reflections = self.reflect(input_rays)

        ax.quiver(self.points[0],self.points[1],self.points[2],reflections[0],reflections[1],reflections[2], length=1, normalize=True)
        ax.quiver(self.points[0],self.points[1],self.points[2],input_rays[0],input_rays[1],input_rays[2], length=1, normalize=True,color='orange')

    def collect(self, mirror, input_rays=None):
        if mirror.focus is not None:
            # draw a line from each point on the collector to the focus
            dim, width, height = self.points.shape
            mdim, mwidth, mheight = mirror.points.shape
            newimg = np.zeros(self.image.shape)
            for ix in range(width):
                for iy in range(height):
                    f = mirror.focus
                    a = 2 * np.sqrt(f) * np.sign(f)
                    x,y,z = self.points[:,ix,iy]
                    kp = (z-f + np.sqrt((z-f)**2 + 4*f*(a*x*x + a*y*y)))/(2*(a*x*x + a*y*y))
                    km = (z-f - np.sqrt((z-f)**2 + 4*f*(a*x*x + a*y*y)))/(2*(a*x*x + a*y*y))
                    ppoint = np.array([kp*x, kp*y, kp*z-kp*f+f]).T
                    mpoint = np.array([km*x, km*y, km*z-km*f+f]).T

                    # mpoint and ppoint are the two points on the paraboloid which
                    # reflect into x,y,z. Now to get their colors and add them up.
                    # just going to use ppoint for now

                    xin = int(np.around(ppoint[0] + mwidth/2 - 0.01))
                    yin = int(np.around(ppoint[1] + mheight/2 - 0.01))
                    newimg[xin,yin] = mirror.image[xin, yin]
            self.image = newimg
        else:
            pass


class Reflector2D():
    def __init__(self, input_fn, light_angle = np.pi/2):
        self.input_fn = input_fn
        self.light_angle = light_angle

    def make_plot(self, minx = -5, maxx = 5, lines  = 20, 
        showfunc=True, shownorms=True, showinrays=True, showoutrays=True):
        '''
        minx - (number) The minimum of the graph
        maxx - (number) The maximum of the graph
        lines - (int) The number of rays to cast

        Shows a pyplot of where rays go for the function of this reflector.
        '''
        in_array, step = np.linspace(minx, maxx, 1000, retstep=True)
        out_array = self.input_fn(in_array)

        fig, ax = plt.subplots()
        plt.axis('equal')

        # plot function itself
        if showfunc:
            #newfunct = lambda x: .25 + 50*np.abs(x)**1.99
            ax.plot(in_array, out_array, color="black")
            #ain_array, step = np.linspace(-1, 1, 1000, retstep=True)
            #ax.plot(ain_array, newfunct(ain_array), color="red")

        # plot reflection lines
        if shownorms or showinrays or showoutrays:
            for x in np.linspace(minx, maxx, lines):
                y = self.input_fn(x)
                norm = -1.0/derivative(self.input_fn, x, dx = step/2)
                if shownorms:
                    k = 1/np.linalg.norm((1, norm))
                    ax.plot([x-.5*k,x+.5*k], [y-.5*norm*k,y+.5*norm*k], color="blue")
                if showinrays:
                    # rays in
                    k = 1
                    ax.plot([x-.5*k*np.cos(self.light_angle) , x+.5*k*np.cos(self.light_angle)],[y-.5*k*np.sin(self.light_angle) , y+.5*k*np.sin(self.light_angle)],color="orange")
                if showoutrays:
                    # rays out
                    out_angle = self.reflect((x,y), norm)
                    k = 10
                    ax.plot([x-.5*k*np.cos(out_angle) , x+.5*k*np.cos(out_angle)],[y-.5*k*np.sin(out_angle) , y+.5*k*np.sin(out_angle)],color="green")
        
        plt.show()

    def reflect(self, point, normal=None):
        '''
        point - (x0, y0) point at which the reflection occurs
        normal - angle of ray that is being frelected around
        '''
        x0, y0 = point
        if normal is None:
            normal = -1.0/derivative(self.input_fn, x0, dx = 0.01)

        return 2*np.arctan(normal) - self.light_angle

if __name__ == '__main__':
    image = cv2.imread("gradient_1d.png")
    r = Reflector2D(lambda x: np.abs(x)**2.0, np.pi/3)
    r.make_plot(-2,2,lines=10)
    plt.show()