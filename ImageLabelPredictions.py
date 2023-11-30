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
class ImageLabelPredictions:
    
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
        _, patchHOG = hog(p.getPatch(), orientations=9, pixels_per_cell=self.cellSize, cells_per_block=self.blockSize, visualize=True)
        hog_image_rescaled = rescale_intensity(patchHOG, in_range=(0, 10))
        return hog_image_rescaled
    
    def read_model(self):
        with open('data.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        self.model = loaded_data

    # labels a set of testing images based on user-specified patch size and scale (pyramiding)
    # adds section of predictions and probabilities to overall list
    def thread_method(self, test_images, finalLabels, name):
        labels = []
        count = 0
        for image in test_images:
            count += 1
            scaleCounts, ctrlCounts = [], []
            img = cv2.imread(image.getFile())
            img = cv2.resize(img, (image.getSize(1), image.getSize(0)))
            print(f"thread {name}: {count}/{len(test_images)}")
            for scale in self.scales:
                image.size = self.imageSize
                imgBordered = img.copy()
                borderWidth, borderHeight = int(self.patchSize[0]*scale*(1-self.stepSize)), int(self.patchSize[1]*scale*(1-self.stepSize))
                imgBordered = cv2.copyMakeBorder(imgBordered, borderHeight, borderHeight, borderWidth, borderWidth, cv2.BORDER_CONSTANT, None, value=[0, 0, 0])
                image.size = list(imgBordered.shape)[:-1]
                image.image = imgBordered
                a, b = int(self.patchSize[0] * scale), int(self.patchSize[1] * scale)
                scaleCount, ctrlCount = 0, 0
                imgActivePatch = imgBordered.copy()
                for i in range(0, image.getSize(0) - b + 1, int(b * self.stepSize)):
                    for j in range(0, image.getSize(1) - a + 1, int(a * self.stepSize)):
                        p = Patch()
                        patch = image.getImage(a, b, i, j)
                        p.size = (list(patch.shape)[1], list(patch.shape)[0])
                        patch = cv2.resize(patch, self.patchSize)                            
                        p.patch, p.prediction, p.probability = patch, None, None
                        prediction, probability = p.label(self.model, self.cellSize, self.blockSize)
                        # imgActivePatch = imgBordered.copy()
                        if (p.prediction == 1 and p.probability[1] > 0.75):
                            color = (0, 255, 0)
                            imgActivePatchTemp = cv2.rectangle(imgActivePatch, (j, i), (j + a, i + b), color=color, thickness=4)
                            imgActivePatchTemp = cv2.resize(imgActivePatchTemp, self.patchSize)
                        # del imgActivePatch
                        if prediction == 1:
                            scaleCount += 1
                        ctrlCount += 1
                        image.update(prediction, probability)
                if True:
                    _, (ax1) = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
                    ax1.axis('off')
                    ax1.set_title(f'Testing image with patch {ctrlCount}: {(list(imgBordered.shape)[:-1][1], list(imgBordered.shape)[:-1][0])}, buffer: {(borderWidth, borderHeight)}, start: {(i, j)}, end: {(i + a, j + b)}')
                    ax1.imshow(imgActivePatchTemp, cmap=plt.cm.gray)
                    plt.show()
                del imgBordered
                scale_pred = image.get_prediction(self.imageRatio)
                scale_prob = image.get_probability(self.imageRatio)
                print(f"{scale_pred} - {scale_prob}%\n")
                scaleCounts.append(scaleCount)
                ctrlCounts.append(ctrlCount)
            labels.append([image.get_prediction(self.imageRatio), image.get_probability(self.imageRatio)])
            print(f"thread {name}: {count}/{len(test_images)} (Prediction: {image.get_prediction(self.imageRatio)}, Probability: {image.get_probability(self.imageRatio)}, counts: {scaleCounts}, ctrl: {ctrlCounts})")
            if self.displayTest:
                image.display(name, count, len(test_images), image.get_prediction(self.imageRatio), image.get_probability(self.imageRatio), scaleCounts, ctrlCounts)
        finalLabels.append(labels)
    
    # utilize multiple treads to parallel as they each cover a set of the testing images
    # returns a list of the predictions and probabilities for all testing images
    def get_labels(self, testingImages):
        self.read_model()
        finalLabels = []
        threadImageIndex = round(len(testingImages)/4)

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Threads have started.")
        plt.show() 

        self.thread_method(testingImages, finalLabels, 1)

        # t1 = Thread(target=self.thread_method, args=(testingImages[:threadImageIndex], finalLabels, 1), name="1")
        # t1.start()
        # t2 = Thread(target=self.thread_method, args=(testingImages[threadImageIndex:2*threadImageIndex], finalLabels, 2), name="2")
        # t2.start()
        # t3 = Thread(target=self.thread_method, args=(testingImages[2*threadImageIndex:3*threadImageIndex], finalLabels, 3), name="3")
        # t3.start()
        # t4 = Thread(target=self.thread_method, args=(testingImages[3*threadImageIndex:4*threadImageIndex], finalLabels, 4), name="4")
        # t4.start()
        # t5 = Thread(target=self.thread_method, args=(testingImages[4*threadImageIndex:5*threadImageIndex], finalLabels, 5), name="5")
        # t5.start()
        # t6 = Thread(target=self.thread_method, args=(testingImages[5*threadImageIndex:6*threadImageIndex], finalLabels, 6), name="6")
        # t6.start()
        # t7 = Thread(target=self.thread_method, args=(testingImages[6*threadImageIndex:7*threadImageIndex], finalLabels, 7), name="7")
        # t7.start()
        # t8 = Thread(target=self.thread_method, args=(testingImages[7*threadImageIndex:], finalLabels, 8), name="8")
        # t8.start()
        # t1.join()
        # t2.join()
        # t3.join()
        # t4.join()
        # t5.join()
        # t6.join()
        # t7.join()
        # t8.join()

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Threads have ended.")
        plt.show() 
        return finalLabels
    
    # read in testing images, equalize the brightness and add borders to said images
    # returns a list of equally bright and padded images
    def get_testing_images(self):
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Gathering testing images.")
        plt.show() 

        testingImages = []
        for file in glob.iglob('testing\\*'):
            i = Image()
            img = cv2.imread(file)
            img = cv2.resize(img, self.imageSize)
            i.file, i.size, i.image, i.predictions, i.probabilities = file, list(img.shape)[:-1], img, [], []
            testingImages.append(i)
        return testingImages
        
    # write the testing images' predictions and probabilities into a text file
    def results(self):
        testingImages = self.get_testing_images()
        labels = self.get_labels(testingImages)
        outputFile = open(self.directory + "/results.txt", "w")
        for testImage, label in zip(testingImages, labels):
            outputFile.write(f"{testImage.get_name()}: \n")
            outputFile.write(f"\tPrediction- {label[0]}\n")
            outputFile.write(f"\tProbability- {label[1]}\n")
        outputFile.close()

    # run the Image Processor 
    def main(self):
        self.results()
