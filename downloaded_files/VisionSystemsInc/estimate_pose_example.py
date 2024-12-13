""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import numpy as np
import os
from PIL import Image, ImageFile
import pix2face.test
import pix2face_estimation.camera_estimation

# Set this to an integer value to run on a CUDA device, None for CPU.
cpu_only = int(os.environ.get("CPU_ONLY")) != 0
cuda_device = None if cpu_only else 0
if cpu_only:
    print("Running on CPU")
else:
    print("Running on cuda device %s" % cuda_device)


ImageFile.LOAD_TRUNCATED_IMAGES = True

this_dir = os.path.dirname(__file__)
img_fname = os.path.join(this_dir, '../pix2face_net/data', 'CASIA_0000107_004.jpg')
img = np.array(Image.open(img_fname))

# create a list of identical images for the purpose of testing timing
num_test_images = 10
imgs = [img,] * num_test_images

# Use dense alignment to estimate pose
pix2face_net = pix2face.test.load_pretrained_model(cuda_device)

import time
t0 = time.time()

# estimate pose for all images in the list
for img in imgs:
    pose = pix2face_estimation.camera_estimation.estimate_head_pose(img, pix2face_net, cuda_device)

t1 = time.time()
total_elapsed = t1 - t0

print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f' % pose)
print('Total Elapsed = %0.1f s : Average %0.2f s / image' % (total_elapsed, total_elapsed / num_test_images))
