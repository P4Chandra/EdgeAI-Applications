'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from basemodel import BaseModel

class FaceDetection(BaseModel):
    '''
    Class for the Face Detection Model.
    '''
   
    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_dict = self.preprocess_input(image)
        #initiate a request
        outputs=self.net.start_async(request_id=0,inputs=input_dict)

        #wait for response,process outputs and return final coordinates and image
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs[self.output_name]
            result_coords,image,cropped_image= self.preprocess_output(outputs,image)
            return cropped_image,image,result_coords
        

    
    def preprocess_output(self, outputs,image):
    # Implemented improvement to detect the most prominent face if more than one face exists
        best_coords=[]
        best_face_area =0.0
        cropped_face=None
        for box in outputs[0][0]:
            conf=box[2]
            if conf > self.threshold :
                xmin=int(box[3]*self.initial_w)
                ymin=int(box[4]*self.initial_h)
                xmax=int(box[5]*self.initial_w)
                ymax=int(box[6]*self.initial_h)
                current_face_area=(xmax-xmin)*(ymax-ymin)
                if current_face_area > best_face_area:
                    best_face_area=current_face_area
                    result_coords=[xmin,ymin,xmax,ymax]
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (234,30,110),2)
                    cropped_face=image[ymin:ymax,xmin:xmax]
        return result_coords,image,cropped_face   
