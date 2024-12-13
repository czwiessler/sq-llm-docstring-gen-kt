import tensorflow as tf
import numpy as np
from PIL import Image
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from crf_rnn_layer import crf_rnn_layer

unaries = tf.expand_dims(tf.Variable(np.array(Image.open('permutohedral_lattice/Images/input.bmp'))/255.0, dtype=tf.float32), axis=0)
reference_image = unaries

num_classes = 3
theta_alpha = 8
theta_beta = 0.125
theta_gamma = 1
num_iterations = 5
output = crf_rnn_layer(unaries, reference_image, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    o = np.round(sess.run(output) * 255).astype(np.uint8)

im = Image.fromarray(np.squeeze(o))
im.save('output.bmp')
print('done')
