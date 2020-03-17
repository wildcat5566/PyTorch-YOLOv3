#!/bin/bash

NUM_CLASSES=$1

echo "
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

# 0
[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

# 1
[maxpool]
size=2
stride=2

# 2
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# 3
[maxpool]
size=2
stride=2

# 4
[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

# 5
[maxpool]
size=2
stride=2

# 6
[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

# 7
[maxpool]
size=2
stride=2

# 8
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

# 9
[maxpool]
size=2
stride=2

# 10
[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

# 11
[maxpool]
size=2
stride=1

# 12
[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

###########

# 13
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

# 14
[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

# 15
[convolutional]
size=1
stride=1
pad=1
filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
activation=linear



# 16
[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=$NUM_CLASSES
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

# 17
[route]
layers = -4

# 18
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

# 19
[upsample]
stride=2

# 20
[route]
layers = -1, 8

# 21
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

# 22
[convolutional]
size=1
stride=1
pad=1
filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
activation=linear

# 23
[yolo]
mask = 1,2,3
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=$NUM_CLASSES
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

" >> yolov3-tiny-custom.cfg
