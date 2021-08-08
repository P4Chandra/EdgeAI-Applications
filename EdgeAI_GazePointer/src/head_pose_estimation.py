'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
#!/usr/bin/env python -W ignore::DeprecationWarning

import cv2
from basemodel import BaseModel

class HeadPoseEstimation(BaseModel):
    '''
    Class for the general Model.
    '''
   
    def predict(self,croppedimage):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_dict = self.preprocess_input(croppedimage)
        #initiate a request
        outputs=self.net.start_async(request_id=0,inputs=input_dict)

        #wait for response,process outputs and return final coordinates and image
        if self.net.requests[0].wait(-1) == 0:
            yaw = self.net.requests[0].outputs['angle_y_fc'][0][0]
            pitch = self.net.requests[0].outputs['angle_p_fc'][0][0]
            roll = self.net.requests[0].outputs['angle_r_fc'][0][0]
            return [yaw,pitch,roll]
        



 
    
       
