from Bridge import *
import sys

# this file is meant to appropriately run the terminal command
args = sys.argv[1:]

with open('args.pkl', 'wb') as file:
            pickle.dump(args, file)

bridge = Bridge(args)
bridge.process()
