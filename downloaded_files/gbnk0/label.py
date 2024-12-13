# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf

class Classify(object):

    def __init__(self, **kwargs):
        self.graph_file = kwargs.get('graph')
        self.graph = self.load_graph(self.graph_file)

    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())

        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def read_tensor_from_image_file(self, file_name,
                                  input_height=299,
                                  input_width=299,
                                  input_mean=0,
                                  input_std=255):
        input_name = "file_reader"

        file_reader = tf.read_file(file_name, input_name)

        
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")

        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def run(self, **kwargs):
        final_results = []
        file_name = kwargs.get('filename')
        label_file = kwargs.get('labels')
        input_layer = kwargs.get('input_layer')
        output_layer = kwargs.get('output_layer')

        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255

        t = self.read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        with tf.Session(graph=self.graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
            results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels(label_file)

        for i in top_k:
            data = {
                "label": labels[i],
                "accuracy": round(float(results[i]) * 100, 2)
            }
            final_results.append(data)
    
        return final_results
