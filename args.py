# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_enabled", type=int, default=1)
parser.add_argument("--train_image_dir", type=str, default="../../train_images")
parser.add_argument("--test_image_dir", type=str, default="../../test_images")
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--vis_dir", type=str, default="visualization")

# for semi-supervised learning
parser.add_argument("--n_labeled_data", type=int, default=100)
args = parser.parse_args()