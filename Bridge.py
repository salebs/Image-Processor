from ImageLabelModel import *

class Bridge:
    def __init__(self, args):
        self.args = args
    
    def isTrue(self, s):
        if (s == "true"):
            return True
        return False

    def process(self):
        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Set up started.")
        plt.show() 

        directory = self.args[0]
        modelType = self.args[1]
        negativePatchCount = int(self.args[2])
        imageSize = (int(self.args[3]), int(self.args[4]))
        patchSize = (int(self.args[5]), int(self.args[6]))
        cellSize = (int(self.args[7]), int(self.args[8]))
        blockSize = (int(self.args[9]), int(self.args[10]))
        stepSize = float(self.args[11])
        scales = [float(self.args[12]), 1, float(self.args[13])]
        displayTrain = self.isTrue(self.args[14])
        displayTest = self.isTrue(self.args[15])

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Set up ended. Process started.")
        plt.show() 

        model = ImageLabelModel(directory, modelType, negativePatchCount, imageSize, patchSize, cellSize, blockSize, scales, stepSize, displayTrain, displayTest)       
        model.main()

        _, (ax1) = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
        ax1.set_title("Process ended.")
        plt.show() 
        return

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
