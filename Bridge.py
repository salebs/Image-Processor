from ImageLabelModel import *
from ImageLabelPredictions import *

# This class handles the connection between the Python backend and the Java frontend.
class Bridge:

    # The constructor has the arguments of the Python terminal command.
    # Every attribute is defined by user in Java Gui.
    def __init__(self, args):
        self.args = args
    
    # determines if string is "true" or "false"
    # returns corresponding boolean
    def isTrue(self, s):
        if (s == "true"):
            return True
        return False

    # process the testing images with the appropriate parameters
    def process(self):
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Set up started.")
        plt.show() 

        modelType = self.args[0] 
        negativePatchCount = int(self.args[1])
        imageSize = (int(self.args[2]), int(self.args[3]))
        cellSize = (int(self.args[4]), int(self.args[5]))
        blockSize = (int(self.args[6]), int(self.args[7]))
        stepSize = float(self.args[8])
        scales = [float(self.args[9]), 1, float(self.args[10])]
        displayTrain = self.isTrue(self.args[12])
        displayTest = self.isTrue(self.args[13])
        patchSize = (round(float(self.args[14])), round(float(self.args[15])))
        imageRatio = float(self.args[16])
        neighbor_count = int(self.args[11])

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Set up ended. Process started.")
        plt.show() 

        model = ImageLabelModel(modelType, negativePatchCount, imageSize, imageRatio, patchSize, cellSize, blockSize, scales, stepSize, neighbor_count, displayTrain, displayTest)       
        model.main()

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Process ended.")
        plt.show() 

    def predict(self):

        modelType = self.args[0] 
        negativePatchCount = int(self.args[1])
        imageSize = (int(self.args[2]), int(self.args[3]))
        cellSize = (int(self.args[4]), int(self.args[5]))
        blockSize = (int(self.args[6]), int(self.args[7]))
        stepSize = float(self.args[8])
        scales = [float(self.args[9]), 1, float(self.args[10])]
        displayTrain = self.isTrue(self.args[12])
        displayTest = self.isTrue(self.args[13])
        patchSize = (round(float(self.args[14])), round(float(self.args[15])))
        imageRatio = float(self.args[16])
        neighbor_count = float(self.args[11])

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Predictions started.")
        plt.show() 
        
        model = ImageLabelPredictions(modelType, negativePatchCount, imageSize, imageRatio, patchSize, cellSize, blockSize, scales, stepSize, neighbor_count, displayTrain, displayTest)       
        model.main()

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Predictions ended.")
        plt.show() 

    # analyze the testing images with the appropriate text files
    def analyze_method(self):
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Analysis has begun.")
        plt.show() 

        directory = self.args[0]
        keyFile = open(directory + "\key.txt", "r")
        keyLines = keyFile.readlines()
        keyFile.close()
        key = {}
        for line in keyLines:
            key[line.split(": ")[0]] = line.split(": ")[1][:-1]

        resultFile = open(directory + "/results.txt")
        resultLines = resultFile.readlines()
        resultFile.close()
        result = {}
        for i in range(len(resultLines)):
            result[resultLines[i].split(": ")[0]] = resultLines[i+1].split("- ")[1][:-1]
            i += 2

        accuracyFile = open(directory + "/accuracy.txt", "w+")
        accuracy = []
        for imgName, label in zip(key.keys(), key.values()):
            if (result.get(imgName) == label):
                accuracy.append(1)
                accuracyFile.write(f"{imgName}: Correct\n\tLabel- {result.get(imgName)}\n\tPrediction- {label}\n")
            else:
                accuracy.append(0)
                accuracyFile.write(f"{imgName}: Incorrect\n\tLabel- {result.get(imgName)}\n\tPrediction- {label}\n")
        
        accuracy = sum(accuracy)/len(accuracy)
        accuracyFile.write(f"\nAccuracy = {accuracy}%")
        
        accuracyFile.close()
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Analysis has ended.")
        plt.show() 
        return
