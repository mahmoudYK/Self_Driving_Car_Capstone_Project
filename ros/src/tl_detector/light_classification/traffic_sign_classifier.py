# Reuse of the tensorflow object detection API in this repo: 
# https://github.com/hemingchen/CarND-Capstone-Traffic-Light-Detection
# The model was trained on data from the simulator using the annotated data from:
# https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing 

# I modified the code and make it reusbale for traffic light detection part in the project

from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import tensorflow as tf
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Tensor Flow object detection API should be installed
# under classifier/models
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self):
        self.LABEL_PATH = os.path.abspath(os.getcwd())+'/light_classification/label_map.pbtxt'
        self.SIM_MODEL_PATH = os.path.abspath(os.getcwd())+'/light_classification/frozen_inference_graph.pb'
        self.label_map = label_map_util.load_labelmap(self.LABEL_PATH)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=4, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)


    def load_image_into_numpy_array(self, image):
        return np.asarray(image)#.astype(np.uint8)

    def import_graph(self,model_path):
        detection_graph = tf.Graph()
    
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
    
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        return detection_graph
        

    def predict(self, detection_graph, image):
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
                image_np = self.load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                time0 = time.time()

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                time1 = time.time()

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                min_score_thresh = .80
                max_idx = np.argmax(scores)
                if scores[max_idx] > min_score_thresh:
                    class_name = self.category_index[classes[max_idx]]['name']
                    return class_name
            
        return 'unknown'
                    
               
#Test the TLClassifier class

#tlc = TLClassifier()                    
#model = tlc.import_graph(tlc.SIM_MODEL_PATH) 

#img = cv.imread('green.jpg')
#tlc.predict(model, img)