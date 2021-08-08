'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
#!/usr/bin/env python -W ignore::DeprecationWarning

import cv2
from basemodel import BaseModel

class FacialLandmarkDetection(BaseModel):
    '''
    Class for the Facial Landmark detection Model.
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
            outputs = self.net.requests[0].outputs[self.output_name]
            eye_coord,right_eye,left_eye,image= self.preprocess_output(outputs,croppedimage)
            return eye_coord,right_eye,left_eye,image


    def preprocess_output(self, outputs,image):
    # Implemented improvement to detect the most prominent face if more than one face exists
        eye_size=25
        outputs=outputs[0]
        h,w,c=image.shape
        leyex,leyey =int(outputs[0][0]*w),int(outputs[1][0]*h)
        reyex,reyey =int(outputs[2][0]*w),int(outputs[3][0]*h)

        lx_min,lx_max=leyex-eye_size,leyex+eye_size
        ly_min,ly_max=leyey-eye_size,leyey+eye_size

        rx_min,rx_max=reyex-eye_size,reyex+eye_size
        ry_min,ry_max=reyey-eye_size,reyey+eye_size

        right_eye= image[ry_min:ry_max,rx_min:rx_max]
        left_eye=  image[ly_min:ly_max,lx_min:lx_max]

    
        eye_coord=[[lx_min,ly_min,lx_max,ly_max],
                   [rx_min,ry_min,rx_max,ry_max]]

        return eye_coord,right_eye,left_eye,image
