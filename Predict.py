from Bridge import *

# this file is meant to appropriately run the terminal command
with open('args.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

args = loaded_data
bridge = Bridge(args)
bridge.predict()
