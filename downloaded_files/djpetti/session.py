import os
import random

import cv2

import numpy as np

import tensorflow as tf

import features


class Session(object):
  """ Represents a session. """

  def __init__(self, **kwargs):
    # The path to the directory containing all the frames.
    self.frame_dir = kwargs.get("frame_dir")
    # A list of the paths to all the image files associated with this session.
    # These are relative to frame_dir.
    self.frame_files = kwargs.get("frame_files")
    # Whether each image is valid or not, indicated by 0 or 1 in the list.
    self.valid = kwargs.get("valid")
    # The bbox information for each face.
    self.face_bboxes = kwargs.get("face_bboxes")

    # List of data that should be converted to bytes features. Each item
    # should be a numpy array containing the data for a feature.
    self.bytes_features = kwargs.get("bytes_features", [])
    # List of data that should be converted to float features. Each item
    # should be a numpy array containing the data for a feature.
    self.float_features = kwargs.get("float_features", [])
    # List of data that should be converted to int features. Each item
    # should be a numpy array containing the data for a feature.
    self.int_features = kwargs.get("int_features", [])

  def __shuffle_list(self, to_shuffle, indices):
    """ Shuffles a list in-place.
    Args:
      to_shuffle: The list to shuffle.
      indices: The list of which indices go where. """
    if len(to_shuffle) != len(indices):
      raise RuntimeError("Cannot shuffle list of len %d with %d indices." % \
                         (len(to_shuffle), len(indices)))

    old = to_shuffle[:]
    if type(to_shuffle) == np.ndarray:
      old = np.array(to_shuffle, copy=True)
    for i, index in enumerate(indices):
      to_shuffle[i] = old[index]

  def __extract_face_crop(self, image, face_data):
    """ Extract the face crop from an image.
    Args:
      image: The image to process.
      face_data: The crop data for this image.
    Returns:
      A cropped version of the image. A None value in this
      list indicates a face crop that was not valid. """
    face_x, face_y, face_w, face_h = face_data[:4]

    start_x = int(face_x)
    end_x = start_x + int(face_w)
    start_y = int(face_y)
    end_y = start_y + int(face_h)

    start_x = max(0, start_x)
    end_x = min(image.shape[1], end_x)
    start_y = max(0, start_y)
    end_y = min(image.shape[0], end_y)

    # Crop the image.
    crop = image[start_y:end_y, start_x:end_x]

    # Resize the crop.
    crop = cv2.resize(crop, (400, 400))

    return crop

  def __convert_features(self, raw_features, convert_func):
    """ Converts features from their raw format to the proper TF format.
    Args:
      raw_features: A list of the features to convert.
      convert_func: The function to use to perform the conversion.
    Returns:
      The same list, but converted. """
    for i, feature in enumerate(raw_features):
      raw_features[i] = convert_func(feature)

    return raw_features

  def _load_crop(self, frame, face_bbox):
    """ Loads and crops the face image.
    Args:
      frame: The name of the frame file.
      face_bbox: The face bounding box data. """
    frame_path = os.path.join(self.frame_dir, frame)

    image = cv2.imread(frame_path)
    if image is None:
      raise RuntimeError("Failed to read image: %s" % (frame_path))

    # Extract the crop.
    return self.__extract_face_crop(image, face_bbox)

  def shuffle(self):
    """ Shuffles all examples in the session. """
    indices = range(0, len(self.valid))
    random.shuffle(indices)

    # Shuffle each list.
    self.__shuffle_list(self.frame_files, indices)
    self.__shuffle_list(self.valid, indices)
    self.__shuffle_list(self.face_bboxes, indices)

    for feature in self.bytes_features:
      self.__shuffle_list(feature, indices)
    for feature in self.float_features:
      self.__shuffle_list(feature, indices)
    for feature in self.int_features:
      self.__shuffle_list(feature, indices)

  def get_random(self):
    """ Gets the next random example, assuming they have already been
    shuffled. It raises a ValueError if there are no more examples.
    Returns:
      The next random example, including the image feature, the byte label features,
      the float features, and the int label features. """
    valid = False

    # Skip any frames that are not valid.
    while not valid:
      if len(self.frame_files) == 0:
        raise ValueError("No more examples.")
      frame = self.frame_files.pop()
      end_index = len(self.frame_files)

      # Can't pop from a Numpy array, but we can look at the end.
      bytes_features = [feature[end_index] for feature in self.bytes_features]
      float_features = [feature[end_index] for feature in self.float_features]
      int_features = [feature[end_index] for feature in self.int_features]

      face_bbox = self.face_bboxes[end_index]
      valid = self.valid[end_index]

    # Extract the face crop.
    crop = self._load_crop(frame, face_bbox)
    # Convert to a feature.
    ret, encoded = cv2.imencode(".jpg", crop)
    if not ret:
      raise ValueError("Encoding frame '%s' failed." % (frame))
    crop = features.to_bytes([tf.compat.as_bytes(encoded.tostring())])

    # Convert to the proper TF feature type.
    bytes_features = self.__convert_features(bytes_features,
                                              features.to_bytes)
    float_features = self.__convert_features(float_features,
                                              features.to_floats)
    int_features = self.__convert_features(int_features, features.to_ints)

    return (crop, bytes_features, float_features, int_features)

  def num_valid(self):
    """
    Returns:
      The number of valid images in the session. """
    num_valid = 0
    for image in self.valid:
      if image:
        num_valid += 1

    return num_valid

