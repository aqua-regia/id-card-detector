# Import packages
import os
import sys
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import tensorflow as tf

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

# Path to image
DIRECTORY_NAME = 'test_images'

DIRECTORY_PATH = os.path.join(CWD_PATH, DIRECTORY_NAME)

onlyfiles = [f for f in listdir(DIRECTORY_PATH) if isfile(join(DIRECTORY_PATH, f))]

for filename in onlyfiles:

    try:
        image_path = DIRECTORY_PATH + "/" + filename
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        image1, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                                                 np.squeeze(boxes),
                                                                                 np.squeeze(classes).astype(np.int32),
                                                                                 np.squeeze(scores),
                                                                                 category_index,
                                                                                 use_normalized_coordinates=True,
                                                                                 line_thickness=3,
                                                                                 min_score_thresh=0.60)

        ymin, xmin, ymax, xmax = array_coord
        shape = np.shape(image1)
        im_width, im_height = shape[1], shape[0]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

        print(f"image is:- {filename}")
        # cv2.imshow("original", image)

        if ymin == xmax == xmin == ymax == 0:
            output_image_path = DIRECTORY_PATH + "/ignored/" + filename
            print(f"output image path:- {output_image_path}")
            cv2.imwrite(output_image_path, image1)
            continue
        else:
            output_image_path = DIRECTORY_PATH + "/detected/" + filename
            print(f"output image path:- {output_image_path}")
            cropped_image = image[int(top):int(bottom), int(left):int(right)]
            cv2.imwrite(output_image_path, cropped_image)
            # cv2.imshow('ID CARD DETECTOR', cropped_image)
            cv2.waitKey(0)

    except Exception as e:
        print(e)
        print(f"some exception occured with image:- {DIRECTORY_PATH + '/' + filename}")
        continue

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()
