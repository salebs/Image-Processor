import glob
from itertools import chain
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from threading import Thread

from Image import *
from Patch import *


# This class handles the construction and characteristics of the Image Processor's model and computer vision process.
class ImageLabelModel:
    
    # The constructor has the model's directory, type, negative count, image size, patch size, cell size, block size, scales, step, and whether it will display positive training images or all testing images(or both or neither).
    # Every attribute is read in from arguments in the start.
    def __init__(self, directory, model_type, negative_patch_count, image_size, patch_size, cell_size, block_size, scales, step_size, display_train, display_test):
        self.modelType = model_type
        self.model = None
        self.directory = directory
        self.negativePatchCount = negative_patch_count
        self.imageSize = image_size
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
        _, patchHOG = hog(p.getPatch(), orientations=9, pixels_per_cell=self.cellSize, cells_per_block=self.blockSize, visualize=True, channel_axis=-1)
        hog_image_rescaled = rescale_intensity(patchHOG, in_range=(0, 10))
        return hog_image_rescaled
    
    # read in user-labeled positive patches and obtain rescaled histogram of gradients for each
    # returns a list for the positive patches and a list of the corresponding rescaled histogram of gradients
    def get_positive_patches(self):
        positivePatchHOGs = []
        positivePatches = []
        for file in glob.iglob(f'{self.directory}/positive/*'):
            p = Patch()
            img = cv2.imread(file)
            # B, G, R = cv2.split(img)
            # B = cv2.equalizeHist(B)
            # G = cv2.equalizeHist(G)
            # R = cv2.equalizeHist(R)
            # img = cv2.merge((B, G, R))
            p.size = (len(img[0]), len(img[1]))
            patch = p.pad(img, self.patchSize)
            p.patch, p.prediction, p.probability = patch, 1, [0.0, 1.0]
            hog_image_rescaled = self.rhog(p)
            positivePatchHOGs.append(hog_image_rescaled)
            positivePatches.append(p.patch)
        return positivePatchHOGs, positivePatches

    # extract random patches from an image
    # returns a list of random patches taken from image
    def extract_patches(self, image):
        extractor = PatchExtractor(patch_size=self.patchSize, max_patches=self.negativePatchCount, random_state=0)
        patches = extractor.transform(image[np.newaxis])
        return patches

    # read in user-labeled negative images and obtain rescaled histogram of gradients for each
    # returns a list for the negative patches and a list of the corresponding rescaled histogram of gradients
    def get_negative_patches(self):
        negativePatchHOGs = []
        negativePatches = []
        for file in glob.iglob(f'{self.directory}/negative/*'):
            image = cv2.imread(file)
            image = cv2.resize(image, self.imageSize)
            # B, G, R = cv2.split(image)
            # B = cv2.equalizeHist(B)
            # G = cv2.equalizeHist(G)
            # R = cv2.equalizeHist(R)
            # image = cv2.merge((B, G, R))
            patches = self.extract_patches(image)
            for patch in patches:
                p = Patch()
                p.size, p.patch, p.prediction, p.probability = self.patchSize, patch.astype('uint8'), 0, [1.0, 0.0]
                hog_image_rescaled = self.rhog(p)
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
            model = KNeighborsClassifier(n_neighbors=3)
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

    # read in testing images, equalize the brightness and add borders to said images
    # returns a list of equally bright and padded images
    def get_testing_images(self):
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Gathering testing images.")
        plt.show() 

        testingImages = []
        for file in glob.iglob(f'{self.directory}\\testing\\*'):
            i = Image()
            img = cv2.imread(file)
            img = cv2.resize(img, self.imageSize)
            # B, G, R = cv2.split(img)
            # B = cv2.equalizeHist(B)
            # G = cv2.equalizeHist(G)
            # R = cv2.equalizeHist(R)
            # img = cv2.merge((B, G, R))
            imgPadded = cv2.copyMakeBorder(img, self.patchSize[0], self.patchSize[0], self.patchSize[1],
                                           self.patchSize[1], cv2.BORDER_CONSTANT, None, value=0)
            imgPaddSize = list(imgPadded.shape)[:-1]
            i.file, i.size, i.image, i.predictions, i.probabilities = file, imgPaddSize, imgPadded, [], []
            testingImages.append(i)
        return testingImages

    # labels a set of testing images based on user-specified patch size and scale (pyramiding)
    # adds section of predictions and probabilities to overall list
    def thread_method(self, test_images, finalLabels, name):
        labels = []
        count = 0
        for image in test_images:
            count += 1
            scaleCounts = []
            img = cv2.imread(image.getFile())
            img = cv2.resize(img, (image.getSize(1), image.getSize(0)))
            print(f"thread {name}: {count}/{len(test_images)}")
            for scale in self.scales:
                print(f"thread {name}: {count}/{len(test_images)} (scale: {scale})")
                a, b = int(self.patchSize[0] * scale), int(self.patchSize[1] * scale)
                scaleCount = 0
                for i in range(0,  image.getSize(0) - a + 1, int(a * self.stepSize)):
                    for j in range(0, image.getSize(1) - b + 1, int(b * self.stepSize)):
                        p = Patch()
                        patch = image.getImage(a, b, i, j)
                        p.size = (list(patch.shape)[1], list(patch.shape)[0])
                        patch = cv2.resize(patch, self.patchSize)
                        p.patch, p.prediction, p.probability = patch, None, None
                        prediction, probability = p.label(self.model, self.cellSize, self.blockSize)
                        if prediction == 1:
                            scaleCount += 1
                        image.update(prediction, probability)
                scaleCounts.append(scaleCount)
            labels.append([image.get_prediction(), image.get_probability()])
            print(f"thread {name}: {count}/{len(test_images)} (Prediction: {image.get_prediction()}, Probability: {image.get_probability()}, counts: {scaleCounts})")
            if self.displayTest:
                image.display()
        finalLabels.append(labels)
    
    # utilize multiple treads to parallel as they each cover a set of the testing images
    # returns a list of the predictions and probabilities for all testing images
    def get_labels(self, testingImages):
        self.create_model()
        finalLabels = []
        threadImageIndex = round(len(testingImages)/4)

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Threads have started.")
        plt.show() 

        t1 = Thread(target=self.thread_method, args=(testingImages[:threadImageIndex], finalLabels, 1), name="1")
        t1.start()
        t2 = Thread(target=self.thread_method, args=(testingImages[threadImageIndex:2*threadImageIndex], finalLabels, 2), name="2")
        t2.start()
        t3 = Thread(target=self.thread_method, args=(testingImages[2*threadImageIndex:3*threadImageIndex], finalLabels, 3), name="3")
        t3.start()
        t4 = Thread(target=self.thread_method, args=(testingImages[3*threadImageIndex:4*threadImageIndex], finalLabels, 4), name="4")
        t4.start()
        # t5 = Thread(target=self.thread_method, args=(testingImages[4*threadImageIndex:5*threadImageIndex], finalLabels, 5), name="5")
        # t5.start()
        # t6 = Thread(target=self.thread_method, args=(testingImages[5*threadImageIndex:6*threadImageIndex], finalLabels, 6), name="6")
        # t6.start()
        # t7 = Thread(target=self.thread_method, args=(testingImages[6*threadImageIndex:7*threadImageIndex], finalLabels, 7), name="7")
        # t7.start()
        # t8 = Thread(target=self.thread_method, args=(testingImages[7*threadImageIndex:], finalLabels, 8), name="8")
        # t8.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        # t5.join()
        # t6.join()
        # t7.join()
        # t8.join()

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Threads have ended.")
        plt.show() 
        return finalLabels
        
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
