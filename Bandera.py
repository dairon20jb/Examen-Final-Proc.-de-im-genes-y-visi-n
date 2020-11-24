import math
import numpy as np
import cv2
import os
from hough import *
from orientation_estimate import *
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
class Bandera: # Se crea la clase ColorImage
    def __init__(self,string,image_name): #Constructor definición
        self.string= string # String es la ruta de la imagen
        self.image_name = image_name
        self.image= cv2.imread(os.path.join(self.string,self.image_name)) #Carga la imagen dada la dirección y nombre de la imagen
        self.labels=0
        self.image2= cv2.imread(os.path.join(self.string,self.image_name))
    def Colores(self):
        n_colors = 4
        self.image = np.array(self.image, dtype=np.float64) / 255  # Load Image and transform to a 2D numpy array.
        rows, cols, ch = self.image.shape
        assert ch == 3
        image_array = np.reshape(self.image, (rows * cols, ch))
        print("Fitting model on a small sub-sample of the data")
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        # El met sera kmeans
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        self.labels = model.predict(image_array)
        centers = model.cluster_centers_
        Cantidaddecolores = np.amax(self.labels)
        print("La bandera tiene",Cantidaddecolores + 1, "colores")

    def Porcentaje(self):
        Coloruno = 0
        Colordos = 0
        Colortres = 0
        Colorcuatro = 0
        for x in range(0, len(self.labels)):
            if self.labels[x] == 0:
                Coloruno = Coloruno + 1
            if self.labels[x] == 1:
                Colordos = Colordos + 1
            if self.labels[x] == 2:
                Colortres = Colortres + 1
            if self.labels[x] == 3:
                Colorcuatro = Colorcuatro + 1
        PorcenC1 = (Coloruno * 100) / len(self.labels)
        PorcenC2 = (Colordos * 100) / len(self.labels)
        PorcenC3 = (Colortres * 100) / len(self.labels)
        PorcenC4 = (Colorcuatro * 100) / len(self.labels)
        print("Hay",PorcenC1,"% del color 1")
        print("Hay",PorcenC2,"% del color 2")
        print("Hay",PorcenC3,"% del color 3")
        print("Hay",PorcenC4,"% del color 4")
    def Orientacion(self):
        high_thresh = 300
        bw_edges = cv2.Canny(self.image2, high_thresh * 0.3, high_thresh, L2gradient=True)
        hough1 = hough(bw_edges)
        accumulator = hough1.standard_HT()
        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hough1.find_peaks(accumulator, nhood, acc_thresh, N_peaks)
        string_orien = ""
        horizontal = 0
        vertical = 0
        mixta = 0
        mixta2 = 0
        [_, cols] = self.image2.shape[:2]
        for i in range(len(peaks)):
            theta_ = hough1.theta[peaks[i][1]]
            theta_ = theta_ - 180
            if (np.abs(theta_) <= 95) and ((np.abs(theta_) > 1)):
                horizontal = horizontal + 1
            elif np.abs(theta_) >= 170:
                vertical = vertical + 1
            else:
                if theta_ == 0:
                    mixta = mixta + 1
                else:
                    mixta2 = mixta2 + 1
        if horizontal == 3:
            string_orien = "Horizontal"
        if (vertical == 2) and (mixta < 2):
            string_orien = "Vertical"
        if (horizontal == 2) and (mixta == 2):
            string_orien = "Mixta"
        return (string_orien)