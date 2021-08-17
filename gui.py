from typing import Iterable, List
import matplotlib.pyplot as plt
import numpy as np
from time import time

class GUI():
    def __init__(self, figure, cornersx, cornersy, npoints=4):
        self.gridCornersX = cornersx
        self.gridCornersY = cornersy
        self._figure = figure
        self._npoints = npoints

        self._cid_mouse_press = figure.canvas.mpl_connect("button_press_event",
                                                          self._onMousePress)
        self._cid_mouse_release = figure.canvas.mpl_connect("button_release_event",
                                                            self._onMouseRelease)

        self._axes = self._addSubplot()


        self._leftMousePressLastTime = -1
        plt.draw()

    def _addSubplot(self) -> None:
        ax = self._figure.add_subplot(111)
        return ax
    
    def _onMousePress(self, event) -> None:
        if event.button == 1:
            self._leftMousePressLastTime = time()

    def _onMouseRelease(self, event) -> None:
        if event.button != 1:
            return
        
        if event.inaxes != self._axes:
            return
        
        dt = time() - self._leftMousePressLastTime
        click = dt < 0.5
        if click:
            x, y = event.xdata, event.ydata
            self.gridCornersX.append(x)
            self.gridCornersY.append(y)

            self._axes.plot(x, y, 'ro')
            plt.draw()

        if len(self.gridCornersX) == self._npoints:
            self._figure.canvas.mpl_disconnect(self._cid_mouse_press)
            self._figure.canvas.mpl_disconnect(self._cid_mouse_release)
            
    
    def drawRectangle(self, xcorners: Iterable, ycorners: Iterable):
        plt.plot(xcorners, ycorners, 'g')
        plt.plot([xcorners[-1], xcorners[0]], 
                 [ycorners[-1], ycorners[0]], 'g')

        plt.draw()
    
    def drawPoints(self, x: Iterable, y: Iterable, color: str = 'b') -> None:
        plt.scatter(x, y, color=color)
        plt.draw()
    
    def drawGridsLines(self, grids1_a_4: List[np.ndarray], grid5: List[np.ndarray], color: str = 'g') -> None:
        cont = 1 
        for grid in grids1_a_4:
            print("nao estou na ultima grid")
            y_size, x_size, *_ = grid.shape

            print(" grid = " , grid)
            print("y_size = ", y_size)
            print("x_size = ", x_size)

            for i in range(y_size):
                print("i dentro do drawGridsLines = ", i)
                plt.plot(grid[i, ..., 0], grid[i, ..., 1], color=color)

            for j in range(x_size):
                print("j dentro do drawGridsLines = ", j)
                plt.plot(grid[..., j, 0], grid[..., j, 1], color=color)

#        for grid in grid5:
#            print("nao estou na ultima grid")
#            y_size, x_size, *_ = grid.shape

#            print(" grid = " , grid)
#            print("y_size = ", y_size)
#            print("x_size = ", x_size)

#            for i in range(y_size):
#                print("i dentro do drawGridsLines = ", i)
#                plt.plot(grid[i, ..., 0], grid[i, ..., 1], color=color)

#            for j in range(x_size):
#                print("j dentro do drawGridsLines = ", j)
#                plt.plot(grid[..., j, 0], grid[..., j, 1], color=color)


    def imshow(self, img: np.ndarray) -> None:
        self._axes.imshow(img)

    def pause(self):
        plt.pause(0.016)

    def block(self):
        plt.show()

