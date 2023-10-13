import cv2
import numpy as np
from skimage.feature import hog


class Patch:
    def __int__(self):
        self.patch = None
        self.size = (0, 0)
        self.prediction = None
        self.probability = None

    def pad(self, img, patch_size):
        height, width = self.size
        desiredWidth, desiredHeight = patch_size
        scaleWidth, scaleHeight = desiredWidth / width, desiredHeight / height
        validScale = min(scaleWidth, scaleHeight)
        patch = cv2.resize(img, (int(width * validScale), int(height * validScale)))
        paddedPatch = cv2.copyMakeBorder(patch, patch_size[0] - int(height * validScale), 0,
                                         patch_size[1] - int(width * validScale), 0, cv2.BORDER_CONSTANT, None, value=0)
        return paddedPatch

    def label(self, model, cell_size, block_size):
        patchHOG = hog(self.patch, pixels_per_cell=cell_size, cells_per_block=block_size, channel_axis=2)
        prediction = model.predict(np.array(patchHOG, ndmin=2))
        probability = model.predict_proba(np.array(patchHOG, ndmin=2))
        self.prediction = prediction[0]
        self.probability = probability[0]
        return prediction[0], probability[0]
