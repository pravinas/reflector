import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt

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
    r = Reflector2D(lambda x: np.abs(x)**2.0, np.pi/3)
    r.make_plot(-2,2,lines=10)
    plt.show()