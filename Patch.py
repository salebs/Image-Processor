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

    # pads patch's pixel data to achieve uniform size while not stretching the elements (only used for user-labeled positive patches)
    # returns an image
    def pad(self, img, patch_size):
        height, width = self.size
        desiredWidth, desiredHeight = patch_size
        scaleWidth, scaleHeight = desiredWidth / width, desiredHeight / height
        validScale = min(scaleWidth, scaleHeight)
        patch = cv2.resize(img, (int(width * validScale), int(height * validScale)))
        paddedPatch = cv2.copyMakeBorder(patch, patch_size[0] - int(height * validScale), 0,
                                         patch_size[1] - int(width * validScale), 0, cv2.BORDER_CONSTANT, None, value=0)
        return paddedPatch

    # labels the individual patch based on previously trained model
    # returns appropriate float representations of the patch's probability and prediction
    def label(self, model, cell_size, block_size):
        _, patchHOG = hog(self.patch, orientations=9, pixels_per_cell=cell_size, cells_per_block=block_size, visualize=True, channel_axis=-1)
        hog_image_rescaled = rescale_intensity(patchHOG, in_range=(0, 10))
        nx, ny = hog_image_rescaled.shape
        hog_image_rescaled = hog_image_rescaled.reshape((1, nx*ny))
        prediction = model.predict(hog_image_rescaled)
        probability = model.predict_proba(hog_image_rescaled)
        self.prediction = prediction[0]
        self.probability = probability[0]
        return prediction[0], probability[0]
