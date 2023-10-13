import cv2


class Image:
    def __int__(self):
        self.image = None
        self.file = None
        self.size = None
        self.predictions = []
        self.probabilities = []

    def update(self, prediction, probability):
        self.predictions.append(prediction)
        self.probabilities.append(probability)

    def get_prediction(self):
        if sum(self.predictions) / len(self.predictions) >= 0.5:
            return 1
        return 0

    def get_probability(self):
        if self.get_prediction() == 1:
            posProb = [probability[1] for probability in self.probabilities]
            validPosProb = [posProb[i] for i in range(0, len(self.predictions)) if self.predictions[i] == 1]
            return sum(validPosProb) / len(validPosProb)
        negProb = [probability[0] for probability in self.probabilities]
        validNegProb = [negProb[i] for i in range(0, len(self.predictions)) if self.predictions[i] == 0]
        return sum(validNegProb) / len(validNegProb)

    def display(self):
        img = cv2.imread(self.file)
        img = cv2.resize(img, self.size)
        cv2.imshow(f"Prediction: {self.get_prediction()}, Probability: {self.get_probability()}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_name(self):
        return self.file[self.file.rfind("/") + 1:self.file.rfind(".")]

