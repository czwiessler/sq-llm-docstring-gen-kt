""" This Script Demonstrates the basic image -> PNCC + offsets --> coefficient estimation --> 3D Jitter pipeline. """

import numpy as np
import os
from PIL import Image, ImageFile
import face3d
import vxl
import pix2face
import glob
import skimage.external.tifffile as tifffile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('output_dir')
args = parser.parse_args()


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Estimate PNCC and Offsets using pix2face network

data_dir = args.input_dir
output_dir = args.output_dir
img_filenames = glob.glob(data_dir + '/*.jpg')

this_dir = os.path.dirname(__file__)
pix2face_data_dir = os.path.join(this_dir, '../pix2face/data/')

model_fname = os.path.join(pix2face_data_dir, 'models/pix2face_unet_v10.pt')
model = pix2face.test.load_model(model_fname)


for img_fname in img_filenames:
    img = np.array(Image.open(img_fname))
    print('Estimating PNCC + Offsets..')
    outputs = pix2face.test.test(model, [img,])
    pncc = outputs[0][0]
    offsets = outputs[0][1]
    print('..Done')

    pvr_data_dir = os.path.join(this_dir, '../face3d/data_3DMM/')
    debug_dir = ''
    debug_mode = False

    num_subject_coeffs = 199  # max 199
    num_expression_coeffs = 29  # max 29

    # load needed data files
    head_mesh = face3d.head_mesh(pvr_data_dir)
    subject_components = np.load(os.path.join(pvr_data_dir, 'pca_components_subject.npy'))
    expression_components = np.load(os.path.join(pvr_data_dir, 'pca_components_expression.npy'))
    subject_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_subject.npy'))
    expression_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_expression.npy'))

    # keep only the PCA components that we will be estimating
    subject_components = vxl.vnl.matrix(subject_components[0:num_subject_coeffs,:])
    expression_components = vxl.vnl.matrix(expression_components[0:num_expression_coeffs,:])
    subject_ranges = vxl.vnl.matrix(subject_ranges[0:num_subject_coeffs,:])
    expression_ranges = vxl.vnl.matrix(expression_ranges[0:num_expression_coeffs,:])

    # create coefficient estimator
    coeff_estimator = face3d.media_coefficient_from_PNCC_and_offset_estimator(head_mesh, subject_components, expression_components, subject_ranges, expression_ranges, debug_mode, debug_dir)

    basename = os.path.basename(os.path.splitext(img_fname)[0])
    output_basename = os.path.join(output_dir, basename)

    # Estimate Coefficients from PNCC and Offsets
    print('Estimating Coefficients..')
    img_ids = [basename,]
    coeffs, result = coeff_estimator.estimate_coefficients_perspective(img_ids, [pncc,], [offsets,])
    if not result.success:
        print('ERROR estimating coefficients for ' + img_fname)
        continue
    print('..Done.')

    coeffs.save(output_basename + '_coeffs.txt')
    tifffile.imsave(output_basename + '_PNCC.tiff', pncc)
    tifffile.imsave(output_basename + '_offsets.tiff', offsets)
    img3d = pncc + offsets
    tifffile.imsave(output_basename + '_3d.tiff', img3d)
