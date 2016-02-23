# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, default="../images")
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--visualization_dir", type=str, default="visualization")
parser.add_argument("--load_epoch", type=int, default=0)
args = parser.parse_args()