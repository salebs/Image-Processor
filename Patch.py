import cv2
import numpy as np
from skimage.feature import hog
from skimage.exposure import rescale_intensity


# This class handles the characteristics of each patch and enables us to perform patch specific methods.
class Patch:
    
    # The constructor has the patch's pixel data, size, prediction, and probability.
    # Every attribute is empty in the start.
    def __int__(self):
        self.patch = None
        self.size = (0, 0)
        self.prediction = None
        self.probability = None

    # labels the individual patch based on previously trained model
    # returns appropriate float representations of the patch's probability and prediction
    def label(self, model, cell_size, block_size):
        _, patchHOG = hog(self.patch, orientations=8, pixels_per_cell=cell_size, cells_per_block=block_size, visualize=True)
        hog_image = rescale_intensity(patchHOG, in_range=(0, 10))
        nx, ny = hog_image.shape
        hog_image_rescaled = hog_image.reshape((1, nx*ny))
        prediction = model.predict(hog_image_rescaled)
        probability = model.predict_proba(hog_image_rescaled)
        self.prediction = prediction[0]
        self.probability = probability[0]
        return prediction[0], probability[0]

    def getPatch(self):
        return self.patch
