import glob
from itertools import chain
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from threading import Thread
import pickle

from Image import *
from Patch import *


# This class handles the construction and characteristics of the Image Processor's model and computer vision process.
class ImageLabelModel:
    
    # The constructor has the model's directory, type, negative count, image size, patch size, cell size, block size, scales, step, and whether it will display positive training images or all testing images(or both or neither).
    # Every attribute is read in from arguments in the start.
    def __init__(self, model_type, negative_patch_count, image_size, image_ratio, patch_size, cell_size, block_size, scales, step_size, neighbor_count, display_train, display_test):
        self.modelType = model_type
        self.model = None
        self.neighbor = neighbor_count
        self.negativePatchCount = negative_patch_count
        self.imageSize = image_size
        self.imageRatio = image_ratio
        self.patchSize = patch_size
        self.cellSize = cell_size
        self.blockSize = block_size
        self.scales = scales
        self.stepSize = step_size
        self.displayTrain = display_train
        self.displayTest = display_test

    # visualize a patch against its histogram of gradients
    def visualize(self, patch, hog_image_rescaled):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(patch, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    # obtain the rescaled histogram of gradient for patch
    # returns the pixel data of the rescaled histogram of gradients
    def rhog(self, p):
        _, patchHOG = hog(p, orientations=8, pixels_per_cell=self.cellSize, cells_per_block=self.blockSize, visualize=True)
        hog_image_rescaled = rescale_intensity(patchHOG, in_range=(0, 10))
        return hog_image_rescaled
    
    # pads patch's pixel data to achieve uniform size while not stretching the elements (only used for user-labeled positive patches)
    # returns an image
    def pad(self, img):
        height, width = list(img.shape)[:-1]
        desiredWidth, desiredHeight = self.patchSize
        scaleWidth, scaleHeight = desiredWidth / width, desiredHeight / height
        validScale = min(scaleWidth, scaleHeight)
        newWidth, newHeight = int(width * validScale), int(height * validScale)
        padWidth, padHeight = self.patchSize[0] - newWidth, self.patchSize[1] - newHeight
        patch = cv2.resize(img, (newWidth, newHeight))
        paddedPatch = cv2.copyMakeBorder(patch, padHeight, 0, padWidth, 0, cv2.BORDER_CONSTANT, None, value=0)
        return paddedPatch
    
    # read in user-labeled positive patches and obtain rescaled histogram of gradients for each
    # returns a list for the positive patches and a list of the corresponding rescaled histogram of gradients
    def get_positive_patches(self):
        positivePatchHOGs = []
        positivePatches = []
        for file in glob.iglob('positivePatches\\*'):
            p = Patch()
            img = cv2.imread(file)
            p.size = (len(img[0]), len(img[1]))
            patch = self.pad(img)
            p.patch, p.prediction, p.probability = patch, 1, [0.0, 1.0]
            hog_image_rescaled = self.rhog(p.patch)
            positivePatchHOGs.append(hog_image_rescaled)
            positivePatches.append(p.patch)
        return positivePatchHOGs, positivePatches

    # extract random patches from an image
    # returns a list of random patches taken from image
    def extract_patches(self, image, scale):
        extractor = PatchExtractor(patch_size=(round(self.patchSize[1] * scale), round(self.patchSize[0] * scale)), max_patches=self.negativePatchCount, random_state=0)
        patches = extractor.transform(image[np.newaxis])
        return patches

    # read in user-labeled negative images and obtain rescaled histogram of gradients for each
    # returns a list for the negative patches and a list of the corresponding rescaled histogram of gradients
    def get_negative_patches(self):
        negativePatchHOGs = []
        negativePatches = []
        for _ in range(0, self.neighbor):
            empty_patch = np.zeros((self.patchSize[1], self.patchSize[0], 3), dtype=np.ndarray)
            hog_empty_rescaled = self.rhog(empty_patch)
            negativePatches.append(empty_patch)
            negativePatchHOGs.append(hog_empty_rescaled)
        for file in glob.iglob('negative\\*'):
            image = cv2.imread(file)
            image = cv2.resize(image, self.imageSize)
            scaledPatches = []
            for scale in self.scales:
                patches = self.extract_patches(image, scale)
                for patch in patches:
                    resizedPatch = cv2.resize(patch, self.patchSize)
                    scaledPatches.append(resizedPatch)
            for patch in scaledPatches:
                p = Patch()
                p.size, p.patch, p.prediction, p.probability = self.patchSize, patch.astype('uint8'), 0, [1.0, 0.0]
                hog_image_rescaled = self.rhog(p.patch)
                negativePatchHOGs.append(hog_image_rescaled)
                negativePatches.append(p.patch)
        return negativePatchHOGs, negativePatches

    # train a model based on the user-labeled positive and negative images' rescaled historgram of gradients
    # returns a list of the appropriate patch pixel data and label
    def get_trained(self):
        trainedPositivePatches, positivePatches = self.get_positive_patches()
        
        if self.displayTrain:
            for patch, patchHOG in zip(positivePatches, trainedPositivePatches):
                self.visualize(patch, patchHOG)

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Got positive patches.")
        plt.show() 

        trainedNegativePatches, negativePatches = self.get_negative_patches()

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Got negative patches.")
        plt.show() 

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Train data.")
        plt.show()

        xTrain = np.array([im for im in chain(trainedPositivePatches, trainedNegativePatches)])
        nsamples, nx, ny = xTrain.shape
        xTrain = xTrain.reshape((nsamples,nx*ny))
        yTrain = np.zeros(xTrain.shape[0])
        yTrain[:len(positivePatches)] = 1
        
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Trained data.")
        plt.show()

        return xTrain, yTrain

    # creates model specified by user, trains it with appropriate information, and sets Image Processor's model
    def create_model(self):
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Make model.")
        plt.show() 

        if self.modelType == "KNeighborsClassifier":
            model = KNeighborsClassifier(n_neighbors=self.neighbor)
            _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
            ax1.set_title("KNeighborsClassifier model.")
            plt.show() 
        elif self.modelType == "MLPClassifier":
            model = MLPClassifier(random_state=1, max_iter=300)
            _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
            ax1.set_title("MLPClassifier model.")
            plt.show() 
        else:
            grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)
            _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
            ax1.set_title("GridSearchCV model.")
            plt.show() 
            x_train, y_train = self.get_trained()
            grid.fit(x_train, y_train)
            model = grid.best_estimator_
        x_train, y_train = self.get_trained()
        model.fit(x_train, y_train)

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Made model.")
        plt.show() 
        
        self.model = model

        with open('data.pkl', 'wb') as file:
            pickle.dump(model, file)

    # run the Image Processor 
    def main(self):
        self.create_model()
