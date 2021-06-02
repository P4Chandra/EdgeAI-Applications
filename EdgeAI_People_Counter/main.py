"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from random import randint

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# for Linux Platform
CODEC = 0x00000021

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.

 
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,           
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-md", "--mode", type=str, default=None,
                        help="if single_image_mode,save as an output image "
                        )
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client =  mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def process_frame(frame, result, args, width, height,inferencetime):
    '''
    Draw bounding boxes onto the frame. Determine People in frame and  duration they are in the frame.
    '''
    people=0
    for box in result[0][0]: # Output shape
        conf = box[2]
        if conf >=args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            people+=1
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    textonvideo="Inference Time=  " +"{:.2f}".format(inferencetime)+" msec"
    cv2.putText(frame,textonvideo,(50, 50),font, 1,(0, 255, 255),2, 
                cv2.LINE_4)
    return frame,people

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    ### TODO: Loop until stream is over ###

    #out = cv2.VideoWriter('out.mp4', CODEC, 30, (width,height))
    duration=0
    total_count=0
    prev_count=0
    inferenceTime=0.0
    infereceCounter=0
    while cap.isOpened():
    ### TODO: Read from the video capture ###
        
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inferencetime=time.time()
        infer_network.exec_net(p_frame)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            inferencetime = (time.time()-inferencetime)*1000
            ### TODO: Process the output
            frame,current_count = process_frame(frame, result, args, width, height,inferencetime)
            ### TODO: Extract any desired stats from the results ###
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server 
            
            ### Topic "person": keys of "count" and "total" ###
            if(prev_count<current_count): #person enters the frame
                duration =time.time()
                total_count = total_count + current_count - prev_count
                client.publish("person/count", json.dumps({"count": current_count}))
                client.publish("person/total", json.dumps({"total": total_count}))
            ### Topic "person/duration": key of "duration" ###   
            if(prev_count>current_count): # person left the frame current_count is reduced
                duration = time.time()-duration
                # handle glitch when person just about to leave the frame
                #they get counted again for fraction of second
                if(duration<1): 
                    total_count=total_count-1
                else:   
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    client.publish("person/total", json.dumps({"total": total_count}))
                    client.publish("person/count", json.dumps({"count": prev_count}))
                
            prev_count=current_count
            
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
            #out.write(frame)
        ### TODO: Write an output image if `single_image_mode` ###
                    
        if key_pressed == 27:
            break
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
