import sys
from Bridge import *

# this file is meant to appropriately run the terminal command
newArgs = sys.argv[1:]
with open('args.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
finalArgs = newArgs
for arg in loaded_data[len(newArgs):]:
    newArgs.append(arg)

bridge = Bridge(finalArgs)
bridge.process()