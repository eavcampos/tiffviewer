from typing import Tuple
import numpy as np
import math
import cv2
import copy
from time import sleep
import make_numbers_raiz_b5
import make_numbers_sede_p4
import make_numbers_triangulo_b7
import make_numbers_triangulo_p7
import make_numbers_sede_p1

class Cropper():

    def __init__(self, blocks:int, 
                       parcelsx:int, 
                       parcelsy:int, 
                       cornerPoints: np.ndarray = None) -> None:
        self._blocks = blocks
        self._parcelsx = parcelsx
        self._parcelsy = parcelsy
        
        # X coordinates of the corners of the blocks, list of lists: every list is a list of 4 corners
        self.gridCornersX = []
        # Y coordinates of the corners of the blocks, list of lists: every list is a list of 4 corners
        self.gridCornersY = []
        # set of all the points of the grid (sector 1 - 4), list of lists: every list is a block (grid)
        self._grids1_a_4 = []

        # set of all the points of the grid (sector 5), list of lists: every list is a block (grid)
        self._grid5 = []

        self._w = self._h = 0

        if cornerPoints is not None:
            print("dentro do if cornerPoints")
            self.loadCornerPoints(cornerPoints)
        print("saindo do construtor")
    
    def loadCornerPoints(self, cornerPoints: np.ndarray) -> None:
        self.gridCornersX = [x for x in cornerPoints[..., 1]]
        self.gridCornersY = [y for y in cornerPoints[..., 0]]

    def sortCorners(self):
        n = 4*self._blocks
        gridCornersX = []
        gridCornersY = []
        for i in range(0, n, 4):
            y = np.argsort(self.gridCornersY[i:i+4])
            xcorners = copy.copy(self.gridCornersX[i:i+4])
            ycorners = copy.copy(self.gridCornersY[i:i+4])


            tl = np.array([xcorners[y[0]], ycorners[y[0]]])
            corners = np.array([xcorners, ycorners]).T

            v1 = corners[y[1], ...] - tl
            v2 = corners[y[2], ...] - tl

            ang1 = math.atan(v1[1] / v1[0])
            ang2 = math.atan(v2[1] / v2[0])

            if abs(ang1) < abs(ang2):
                tr = corners[y[1], ...]
                bl = corners[y[2], ...]
            else:
                tr = corners[y[2], ...]
                bl = corners[y[1], ...]
            
            br = corners[y[3], ...]
        
            gridCornersX.append([tl[0], tr[0], br[0], bl[0]])
            gridCornersY.append([tl[1], tr[1], br[1], bl[1]])
        
        self.gridCornersX = gridCornersX
        self.gridCornersY = gridCornersY
    
    def calcGridsPoints(self) -> np.ndarray:

        # for every block...
        cont = 1 
        for gridCornersX, gridCornersY in \
            zip(self.gridCornersX, self.gridCornersY):
            dst = np.array([gridCornersX, gridCornersY], dtype="float32")
            dst = dst.T
              # calculating the rectangles from the trapeziums through the transform function
            src = self.getDstTransformPoints(dst)

            # Base transform matrix that links rectangles to their respective trapeziums
            M = cv2.getPerspectiveTransform(src, dst)

############
 #o if abaixo foi criado para tratar o setor 5, cujas dimensoes sao 5X9, tranposto aos demais, que são 9X5
            if cont == 3:  #TROCAR POR 5, QUE É O ULTIMO SETOR DA P8
               # src[1,0] is width of the resulting rectangle (block)
               x_len = src[1,0] / self._parcelsy # ====> divido pela qtd de parcelas y pq é transposto
               # src[3,1] is height of the resulting rectangle (block)
               y_len = src[3,1] / self._parcelsx # ====> divido pela qtd de parcelas x pq é transposto
               grid = np.zeros((self._parcelsy+1, self._parcelsx+1, 2))
               for y in range(self._parcelsy+1):
                   print("y = ", y) 
                   for x in range(self._parcelsx+1):
                 # through the known points of the transformed rectangle, calculates the points in the corresponding rectangle using the M matrix
                       point = M @ np.array([y*x_len, x*y_len, 1], dtype="float32")
                    # divides by homogeneous coordinate to get the real coordinates
                       grid[y, x] = point[:2] / point[2]
#               self._grid5.append(grid)
            else: #trata os setores 1, 2, 3 e 4
               # src[1,0] is width of the resulting rectangle (block)
               x_len = src[1,0] / self._parcelsx 
               # src[3,1] is height of the resulting rectangle (block)
               y_len = src[3,1] / self._parcelsy 
               grid = np.zeros((self._parcelsy+1, self._parcelsx+1, 2))
               for y in range(self._parcelsy+1):
                   print("y = ", y) 
                   for x in range(self._parcelsx+1):
                 # through the known points of the transformed rectangle, calculates the points in the corresponding rectangle using the M matrix
                       point = M @ np.array([x*x_len, y*y_len, 1], dtype="float32")
                    # divides by homogeneous coordinate to get the real coordinates
                       grid[y, x] = point[:2] / point[2]
            self._grids1_a_4.append(grid)
            cont = cont + 1 

        return self._grids1_a_4, self._grid5
    
    def getDstTransformPoints(self, srcPoints) -> np.ndarray:
        w1 = np.linalg.norm(srcPoints[1, ...] - srcPoints[0, ...])
        w2 = np.linalg.norm(srcPoints[3, ...] - srcPoints[2, ...])
        w = max(w1, w2)

        h1 = np.linalg.norm(srcPoints[3, ...] - srcPoints[0, ...])
        h2 = np.linalg.norm(srcPoints[2, ...] - srcPoints[1, ...])
        h = max(h1, h2)

        tl = np.array([0, 0])
        tr = np.array([w, 0])
        br = np.array([w, h])
        bl = np.array([0, h])

        return np.array([tl, tr, br, bl], dtype="float32")
    
    def getCrop(self, img: np.ndarray, 
                    srcPoints: np.ndarray, 
                    dstPoints: np.ndarray) -> np.ndarray:
        if not (self._w and self._h):
            self._w, self._h = int(srcPoints[2, 0]), int(srcPoints[2, 1])
        crop = np.zeros((self._h, self._w, 3))

        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        for y in range(self._h):
            for x in range(self._w):
                imgPoint = M @ np.array([x, y, 1])
                j, i = imgPoint[:2] / imgPoint[2]
                crop[y, x] = np.flip(img[int(i), int(j)])
        print(" dentro do getCrop - valor crop = ")
        return crop

    def getCrop2(self, img: np.ndarray, 
                    srcPoints: np.ndarray, 
                    dstPoints: np.ndarray) -> np.ndarray:
        if not (self._w and self._h):
            self._w, self._h = int(srcPoints[2, 0]), int(srcPoints[2, 1])
        crop = np.zeros((self._h, self._w, 3))

        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        for y in range(self._h):
            for x in range(self._w):
                imgPoint = M @ np.array([x, y, 1])
                j, i = imgPoint[:2] / imgPoint[2]
#                crop[y, x] = np.flip(img[int(i), int(j)])
                crop[y, x] = img[int(i), int(j)]  #==> deu certo, mas fundo azul
        print(" dentro do getCrop 2 - valor crop = ")
        return crop

    def saveCropsP8(self, img: np.ndarray, 
                          outputdir: str = "",
                          timestamp: str = "", 
                          scale: float = 1) -> None:
        s = scale
        print("Estou no savveCropsP8. valor do scale ou s = ", s)
        for k, g in enumerate(self._grids1_a_4):
            print("K = setor = ", k)
            if k == 2:
               for i in range(self._parcelsy):
                   print(" i = ", i)
                   for j in range(self._parcelsx):
                       print("j = ", j)
#                       print("G[i, j] = " , g[i,j], " g[i, j+1] = ",  g[i, j+1], " g[i+1, j+1] = ", g[i+1, j+1], "g[i+1, j] = ", g[i+1, j])
# scales every point of the block
                       print("TAMAHO DE G = ", len(g))
#                       dst = np.array([s*g[i,j], s*g[i, j+1],  s*g[i+1, j+1], s*g[i+1, j]],dtype="float32")
                       teste = np.array([g[i,j], g[i+1, j],  g[i+1, j+1], g[i, j+1]],dtype="float32")
                       dst = np.array([s*g[i,j], s*g[i+1, j],  s*g[i+1, j+1], s*g[i, j+1]],dtype="float32")
                       print("TESTE = ")
                       print(teste)
                       print("DST = ")
                       print(dst)
                       # gets the corresponding rectangle of the (scaled) trapezium
                       src = self.getDstTransformPoints(dst)
                    # given the coordinates, extracts the crop of the parcel from the original image
                       crop = self.getCrop2(img, src, dst)
                    # to run local tests, replace self.make_numbers_p2 by the function you implemented for your experiment
                       block, number = self.make_numbers_p8(j, i, self._parcelsx, self._parcelsy, k+1)
                       name = outputdir + timestamp + "-" + "%02d" % (block) + "-" + "%03d" % (number) + ".png"
                       print("No K = 1 - name = " , name)
                       cv2.imwrite(name, crop)  

            else:
               for i in range(self._parcelsy):
                   print(" i = ", i)
                   for j in range(self._parcelsx):
                       print("j = ", j)
#                       print("G[i, j] = " , g[i,j], " g[i, j+1] = ",  g[i, j+1], " g[i+1, j+1] = ", g[i+1, j+1], "g[i+1, j] = ", g[i+1, j])
# scales every point of the block 
                       dst = np.array([s*g[i,j], s*g[i, j+1],  s*g[i+1, j+1], s*g[i+1, j]],dtype="float32")
#                       print("DST = ")
#                       print(dst)
                       # gets the corresponding rectangle of the (scaled) trapezium
                       src = self.getDstTransformPoints(dst)
                    # given the coordinates, extracts the crop of the parcel from the original image
                       crop = self.getCrop(img, src, dst)
                    # to run local tests, replace self.make_numbers_p2 by the function you implemented for your experiment
                       block, number = self.make_numbers_p8(i, j, self._parcelsy, self._parcelsx, k+1)
                       name = outputdir + timestamp + "-" + "%02d" % (block) + "-" + "%03d" % (number) + ".png"
                       print("No k = 0 - name = " , name)
                       cv2.imwrite(name, crop)  




    def make_numbers_p8(self, line, col, numLines, numColumns, p8Block: int):

        if p8Block == 3:  # trocar por 5 (ultimo setor da P8)
#           index = (numLines * 5) - (numLines * col) - line
#           index = (numLines * numColumns) - (numLines * col) - line
           index = (numLines* (numColumns-(col+1))) + (line+1)        
#           print("index = (", numLines, " * ", numColumns, ") - ( ", numLines, " * ", col, ") - ", line)
           print("index = ((", numLines, " * (", numColumns, "-(", col, "+1))) + (", line, "+1) ==> ", index)  
        else:
           index = ((numLines - line) * numColumns) - col
#           index = ((numLines - line) * numColumns) - col
#           index = ((line+1) * numColumns) - col
        return p8Block, index



