import cv2
from matplotlib import pyplot as plt


# This class handles the characteristics of each image and enables us to perform image specific methods.
class Image:
    
    # The constructor has the image's pixel data, file, size, prediction list, and probability list.
    # Every attribute is empty in the start.
    def __int__(self):
        self.image = None
        self.file = None
        self.size = None
        self.predictions = []
        self.probabilities = []

    # adds patch's prediction and probability to the images overall prediction and probability lists
    def update(self, prediction, probability):
        self.predictions.append(prediction)
        self.probabilities.append(probability)

    # calculate overall prediction for image
    # returns an integer representing image's prediction
    def get_prediction(self):
        if sum(self.predictions) / len(self.predictions) >= 0.5:
            return 1
        return 0

    # calculate overall probability for image based on prediction
    # returns an integer representing image's probability
    def get_probability(self):
        if self.get_prediction() == 1:
            posProb = [probability[1] for probability in self.probabilities]
            validPosProb = [posProb[i] for i in range(0, len(self.predictions)) if self.predictions[i] == 1]
            return sum(validPosProb) / len(validPosProb)
        negProb = [probability[0] for probability in self.probabilities]
        validNegProb = [negProb[i] for i in range(0, len(self.predictions)) if self.predictions[i] == 0]
        return sum(validNegProb) / len(validNegProb)

    # displays image to user with corresponding prediction and probability
    def display(self, name, count, len_test_images, prediction, probability, scale_counts, ctrl_counts):
        img = cv2.imread(self.file)
        img = cv2.resize(img, (self.size[1], self.size[0]))
        _, (ax1) = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.set_title(f"thread {name}: {count}/{len_test_images} (Prediction: {round(prediction, 2)}, Probability: {round(probability, 2)}, counts: {scale_counts}, ctrl: {ctrl_counts})")
        ax1.imshow(img, cmap=plt.cm.gray)
        plt.show()
        

    # obtain the name of the image based on the file
    # returns a string representation of the image's name
    def get_name(self):
        return self.file[self.file.rfind("\\") + 1:self.file.rfind(".")]

    def getSize(self, i):
        return self.size[i]
    
    def getImage(self, a, b, i, j):
        patch = self.image[i:i + b, j:j + a]
        return patch
    
    def getFile(self):
        return self.file
