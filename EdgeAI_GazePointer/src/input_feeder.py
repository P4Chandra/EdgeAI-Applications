'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray
from face_detection import FaceDetection
import argparse
import time

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.cap=None
        self.input_type=input_type
        self.input_file=None
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        try:
            if self.input_type=='video':
                self.cap=cv2.VideoCapture(self.input_file)
            elif self.input_type=='cam':
                self.cap=cv2.VideoCapture(0)
            else:
                self.cap=cv2.imread(self.input_file)
        except FileNotFoundError:
            print("Cannot locate video file: "+ video_file)
        except Exception as e:
            print("Something else went wrong with the video file: ", e)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        if self.input_type=='image':
            ret,frame=1,self.cap
            yield ret,frame
        else:
            while True:
                for _ in range(10):
                    ret, frame=self.cap.read()
                    yield ret,frame

    def get_image(self):
        return self.cap

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

    def getFPS(self):
        if self.input_type == 'image':
            return 1
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def getVideoLen(self):
        if self.input_type == 'image':
            return 1
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def getVideoWidth(self):
        
        if self.input_type == 'image':
            h,w,_=self.cap.shape
            return w
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def getVideoHeight(self):
        if self.input_type == 'image':
            h,w,_=self.cap.shape
            return h
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
