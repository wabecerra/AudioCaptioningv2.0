#!/bin/bash

# --- Download a pretrained PANN model checkpoint to the local path. ---
# Adjust the path as needed.

CHECKPOINT_PATH="../models/pretrained_weights/Cnn14_mAP=0.431.pth"

wget -O "$CHECKPOINT_PATH" "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
