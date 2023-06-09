import os
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-wcs", "--which_cudas", nargs="+", default=[])
config = parser.parse_args()

if len(config.which_cudas) <= 0:
    raise Exception("Provide a list of cuda device integers using -wcs")
for i in config.which_cudas:
    os.system(f"python pitchmatched_conversion.py -wc {i}")
