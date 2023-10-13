import glob
from itertools import chain
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from Image import *
from Patch import *


class ImageLabelModel:
    def __init__(self):
        self.modelType = ""
        self.model = None
        self.directory = ""
        self.negativePatchCount = 1
        self.imageSize = (0, 0)
        self.patchSize = (0, 0)
        self.cellSize = (2, 2)
        self.blockSize = (2, 2)
        self.scales = (0.5, 1, 2)
        self.stepSize = 0.25
        self.display = False

    def get_positive_patches(self):
        positivePatches = []
        for file in glob.iglob(f'{self.directory}/positive/*'):
            p = Patch()
            img = cv2.imread(file)
            p.size = (len(img[0]), len(img[1]))
            patch = p.pad(img, self.patchSize)
            p.patch, p.prediction, p.probability = patch, 1, [0.0, 1.0]
            positivePatches.append(p)
        return positivePatches

    def extract_patches(self, image):
        extractor = PatchExtractor(patch_size=self.patchSize, max_patches=1, random_state=0)
        patch = extractor.transform(image[np.newaxis])
        return patch

    def get_negative_patches(self):
        negativePatches = []
        for file in glob.iglob(f'{self.directory}/negative/*'):
            image = cv2.imread(file)
            image = cv2.resize(image, self.imageSize)
            for j in range(0, self.negativePatchCount):
                p = Patch()
                patch = self.extract_patches(image)
                p.size, p.patch, p.prediction, p.prediction = self.patchSize, patch[0], 0, [1.0, 0.0]
                negativePatches.append(p)
        return negativePatches

    def train(self):
        positivePatches = self.get_positive_patches()
        negativePatches = self.get_negative_patches()
        xTrain = np.array([hog(p.patch, pixels_per_cell=self.cellSize, cells_per_block=self.blockSize, channel_axis=2)
                          for p in chain(positivePatches, negativePatches)])
        yTrain = np.zeros(xTrain.shape[0])
        yTrain[:len(positivePatches)] = 1
        return xTrain, yTrain

    def create_model(self):
        if self.modelType == "KNeighborsClassifier":
            model = KNeighborsClassifier(n_neighbors=3)
        elif self.modelType == "MLPClassifier":
            model = MLPClassifier(random_state=1, max_iter=300)
        else:
            grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)
            x_train, y_train = self.train()
            grid.fit(x_train, y_train)
            model = grid.best_estimator_
        x_train, y_train = self.train()
        model.fit(x_train, y_train)
        self.model = model

    def get_testing_images(self):
        testingImages = []
        for file in glob.iglob(f'{self.directory}/testing/*'):
            i = Image()
            img = cv2.imread(file)
            img = cv2.resize(img, self.imageSize)
            B, G, R = cv2.split(img)
            B = cv2.equalizeHist(B)
            G = cv2.equalizeHist(G)
            R = cv2.equalizeHist(R)
            img = cv2.merge((B, G, R))
            imgPadded = cv2.copyMakeBorder(img, self.patchSize[0], self.patchSize[0], self.patchSize[1],
                                           self.patchSize[1], cv2.BORDER_CONSTANT, None, value=0)
            i.file, i.size, i.image, i.predictions, i.probabilities = file, self.imageSize, imgPadded, [], []
            testingImages.append(i)
        return testingImages

    def get_labels(self):
        self.create_model()
        testingImages = self.get_testing_images()
        labels = []
        for image in testingImages:  # IMPLEMENT THREADS
            for scale in self.scales:
                a, b = int(self.patchSize[0] * scale), int(self.patchSize[1] * scale)
                for i in range(0,  image.size[0] - a + 1, int(a * self.stepSize)):
                    for j in range(0, image.size[1] - b + 1, int(b * self.stepSize)):
                        p = Patch()
                        patch = image.image[j:j + b, i:i + a]
                        p.patch = cv2.resize(patch, self.patchSize)
                        p.size = (patch.shape[1], patch.shape[0])
                        p.prediction, p.probability = None, None
                        prediction, probability = Patch.label(p, self.model, self.cellSize, self.blockSize)
                        image.update(prediction, probability)
            labels.append([image.get_prediction(), image.get_probability()])
            if self.display:
                image.display()
        return labels

    def results(self):
        testingImages = self.get_testing_images()
        labels = self.get_labels()
        outputFile = open(self.directory + "results.txt", "w")
        for testImage, label in zip(testingImages, labels):
            outputFile.write(f"{testImage.get_name()}: \n")
            outputFile.write(f"\tPrediction: {label[0]}\n")
            outputFile.write(f"\tProbability: {label[1]}\n")
        outputFile.close()

    def main(self):
        self.results()
