from Bridge import *
import sys

# this file is meant to appropriately run the terminal command
args = sys.argv[1:]
bridge = Bridge(args)
bridge.process()
