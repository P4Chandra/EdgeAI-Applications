'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
#!/usr/bin/env python -W ignore::DeprecationWarning

import cv2
import numpy as np
import math
from basemodel import BaseModel

class GazeEstimation(BaseModel):
    '''
    Class for the general Model.
    '''
    
    def predict(self,left_eye,right_eye,head_pose):
        '''
        This method is meant for running predictions on the input image.
        '''
        left_eye_new = self.preprocess_image(left_eye)
        right_eye_new= self.preprocess_image(right_eye)
        inputs_new={'head_pose_angles':head_pose,'left_eye_image':left_eye_new,
                    'right_eye_image':right_eye_new}
        #initiate a request
        self.net.start_async(request_id=0,inputs=inputs_new)

        #wait for response,process outputs and return final coordinates and image
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs[self.output_name]
            mouse_coord=self.preprocess_output(outputs[0],head_pose)
            return outputs[0],mouse_coord 

        

    def preprocess_image(self, image):
        newimage = cv2.resize(image.copy(),(60, 60),interpolation = cv2.INTER_AREA)
        # Change data layout from HWC to CHW
        newimage = newimage.transpose((2,0,1))
        newimage = newimage.reshape(1,*newimage.shape)
        return newimage

    def preprocess_output(self,outputs,head_pose):
        roll=head_pose[2]
        gaze_vector=outputs/cv2.norm(outputs)
        cos=math.cos(roll*math.pi/180)
        sin=math.sin(roll*math.pi/180)
        x=gaze_vector[0]*cos+gaze_vector[1]*sin
        y=-gaze_vector[0]*sin+gaze_vector[1]*cos
        return (x,y)
    
    
