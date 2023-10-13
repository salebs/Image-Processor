from ImageLabelModelMain import *

directory = r'C:\Users\benss\PycharmProjects\COSC 470\OCT\Image Processor'
modelType = "KNeighborsClassifier"
negativePatchCount = 100
imageSize = (1000, 500)
patchSize = (100, 100)
cellSize = (2, 2)
blockSize = (2, 2)
stepSize = 0.25
scales = [3, 2, 1, 0.5]
display = True

ImageLabelModel = ImageLabelModel()
ImageLabelModel.directory = directory
ImageLabelModel.modelType = modelType
ImageLabelModel.negativePatchCount = negativePatchCount
ImageLabelModel.imageSize = imageSize
ImageLabelModel.patchSize = patchSize
ImageLabelModel.cellSize = cellSize
ImageLabelModel.blockSize = blockSize
ImageLabelModel.stepSize = stepSize
ImageLabelModel.scales = scales
ImageLabelModel.display = display

ImageLabelModel.main()
