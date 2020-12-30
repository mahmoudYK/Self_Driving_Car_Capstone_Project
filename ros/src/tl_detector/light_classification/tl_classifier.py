import rospy
import cv2 as cv
import numpy as np
import traffic_sign_classifier as classifier
from styx_msgs.msg import TrafficLight

#import imp
#classifier = imp.load_source('traffic_sign_classifier', './traffic_sign_classifier.py')

class TLClassifier(object):
    def __init__(self):
         # The TLCassifier class is implemented here: ./traffic_sign_classifier.py
         # Tensor Flow object detection API should be installed because it is used by the classifier
         # The trained simulator model is being used assuming that the code will run in the simulator only (corona)
         # For the real world model, plaese use the frozen_inference_graph.pb.real under the same directory
         # For more information plaese refer to the code in ./traffic_sign_classifier.py
         self.tlc = classifier.TLClassifier()                    
         self.model = self.tlc.import_graph(self.tlc.SIM_MODEL_PATH) 
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        im_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        class_name = self.tlc.predict(self.model, im_rgb)
        
        if class_name == 'Red':
            return TrafficLight.RED
        elif class_name == 'Yellow':
            return TrafficLight.YELLOW
        elif class_name == 'Green':
            return TrafficLight.GREEN
        
        return TrafficLight.UNKNOWN
        
        #Old Model
        #image_np = np.asarray(image)
        #if image_np.size == 0 or not (image.shape[0] == 600) or not (image.shape[1] == 800) :
        #    return TrafficLight.UNKNOWN
        #boxes=self.tlc.detect_multi_object(image_np,score_threshold=0.2)
        #if boxes.size == 0:
        #    return TrafficLight.UNKNOWN
        #cropped_image=classifier.crop_roi_image(image_np,boxes[0])
        #plt.imshow(cropped_image)
        #plt.show()
        
        #minidx, color = classifier.classify_color_cropped_image(cropped_image)
        #if minidx == 0:
        #    rospy.loginfo('Red')
        #    return TrafficLight.RED
        #elif minidx == 1:
        #    rospy.loginfo('Yellow')
        #    return TrafficLight.YELLOW
        #elif minidx == 2:
        #    rospy.loginfo('Green')
        #    return TrafficLight.GREEN
        
        #return TrafficLight.UNKNOWN
    

#tlctest = TLClassifier()
#img = cv.imread('dayClip6--00332.jpg')
#print(tlctest.get_classification(img))