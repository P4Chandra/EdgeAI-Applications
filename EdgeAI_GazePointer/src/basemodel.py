'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
from openvino.inference_engine import IENetwork, IECore

class BaseModel:
    '''
    Class for the general Model.
    '''
    def __init__(self, model_name,width,height, device='CPU', extensions=None,threshold=0.60):
        '''
        Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.extensions=extensions
        self.initial_w=width
        self.initial_h=height
        self.net=None
        self.core = None
 
        try:
            self.core = IECore()
            # Add a CPU extension
            if self.extensions and "CPU" in self.device:
                self.core.add_extensions(self.extensions,self.device)

            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.input_info[self.input_name].input_data.shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net=self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h),interpolation = cv2.INTER_AREA)
        image = image.transpose((2,0,1))
        image = image.reshape(1,*image.shape)
        input_dict={self.input_name: image}
        return input_dict

    def check_model(self):
        raise NotImplementedError

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        pass
        
    def preprocess_output(self, outputs,image):
       pass
    
    
