'''
This class can be used to run the actual application,load the models,create output video
Sample usage:
    
'''
import cv2
import argparse
import time
import os
import sys
import logging as log

from numpy import ndarray
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarkDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController

log.basicConfig(filename='debugLog.log', filemode='w', level=log.DEBUG)

def drawEyes(face_coord,eye_coord,frame):
     eyesize=20
     lxmin=eye_coord[0][0]+face_coord[0]
     lymin=eye_coord[0][1]+face_coord[1]
     lxmax=eye_coord[0][2]+face_coord[0]+eyesize
     lymax=eye_coord[0][3]++face_coord[1]+eyesize

     rxmin=eye_coord[1][0]+face_coord[0]
     rymin=eye_coord[1][1]+face_coord[1]
     rxmax=eye_coord[1][2]+face_coord[0]+eyesize
     rymax=eye_coord[1][3]+face_coord[1]+eyesize
     cv2.rectangle(frame,(lxmin,lymin),(lxmax,lymax),(0,0,255),3)
     cv2.rectangle(frame,(rxmin,rymin),(rxmax,rymax),(0,0,255),3)

     reyecenter=[int(rxmin+(rxmax-rxmin)/2),int(rymin+(rymax-rymin)/2)]
     leyecenter=[int(lxmin+(lxmax-lxmin)/2),int(lymin+(lymax-lymin)/2)]
     eye_center_coord=[leyecenter,reyecenter]
  
     return frame,eye_center_coord

def writePoseAngles(frame,pose_coord):
    cv2.putText(frame,"PoseAngles: yaw={:.2f}, pitch={:.2f}, roll={:.2f}".format(
                pose_coord[0],pose_coord[1],pose_coord[2]),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    return frame

def drawGazeAngles(frame,gaze_vector,eye_center_coord):
    x,y=gaze_vector[:2]
    linelen=500
    #draw vector from left eye
    left_eye_center=eye_center_coord[0]
    end_left=(int(left_eye_center[0]+x*linelen),int(left_eye_center[1]-y*linelen))
    frame=cv2.arrowedLine(frame,left_eye_center,end_left,(255,120,120),5)

    #draw vector from right eye
    right_eye_center=eye_center_coord[1]
    end_right=(int(right_eye_center[0]+x*linelen),int(right_eye_center[1]-y*linelen))
    frame=cv2.arrowedLine(frame,right_eye_center,end_right,(255,120,120),5)
    return frame

def main(args):
    model_face=args.model_face
    model_landmark=args.model_landmark
    model_headpose=args.model_headpose
    model_gaze=args.model_gaze
    device=args.device
    input_file=args.input_file
    input_filetype=args.input_filetype
    threshold=args.threshold

    # Initialize Input feeder and load the input file
    feeder=InputFeeder(input_type=input_filetype,input_file=input_file)
    feeder.load_data()
    fps = feeder.getFPS()
    video_len = feeder.getVideoLen()
    initial_w = feeder.getVideoWidth() 
    initial_h = feeder.getVideoHeight()

    #Load all 4 models i.e. Face,head pose,head landmark and gaze
    log.info('Loading Face Detector Model.....')
    start_model_load_time=time.time()
    face= FaceDetection(model_name=model_face,device=device,threshold=threshold,
                      width=initial_w,height=initial_h)
    face.load_model()
    total_model_load_time = time.time() - start_model_load_time
    log.info('Loaded Face Detector Model in {:.2f} secs'.format(total_model_load_time))

    log.info('Loading Face Landmark Detection Model.....')
    start_model_load_time=time.time()
    landmark=FacialLandmarkDetection(model_name=model_landmark,device=device,threshold=threshold,
                      width=initial_w,height=initial_h)
    landmark.load_model()
    total_model_load_time = time.time() - start_model_load_time
    log.info('Loaded Face Landmark Detector Model in {:.2f} secs'.format(total_model_load_time))

    log.info('Loading HeadPose Model.....')
    start_model_load_time=time.time()
    headpose=HeadPoseEstimation(model_name=model_headpose,device=device,threshold=threshold,
                      width=initial_w,height=initial_h)
    headpose.load_model()
    total_model_load_time = time.time() - start_model_load_time
    log.info('Loaded HeadPose Model in {:.2f} secs'.format(total_model_load_time))

    log.info('Loading Gaze Estimation Model.....')
    start_model_load_time=time.time()
    gaze=GazeEstimation(model_name=model_gaze,device=device,threshold=threshold,
                      width=initial_w,height=initial_h)
    gaze.load_model()
    total_model_load_time = time.time() - start_model_load_time
    log.info('Loaded Gaze Estimation Model in {:.2f} secs'.format(total_model_load_time))
    
    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    mousectrl=MouseController("high","fast")

    # define benchmark time parameters
    total_inf_time_facedetection=0.00
    total_inf_time_facelandmark =0.00
    total_inf_time_headpose=0.00
    total_ing_time_gazeest=0.00
    total_time_draw_eyes=0.00
    total_time_gaze_vector=0.00
    total_time_move_ctrl=0.00
    
    log.info('Video stream started....')
    start_inference_time=time.time()
    for ret,frame in feeder.next_batch() :
        counter+=1
        if not ret:
            break
        key = cv2.waitKey(60)
        try:
            start_model_infer_time=time.time()
            face_image,frame,face_coords=face.predict(frame)
            total_inf_time_facedetection += (time.time() - start_model_infer_time)

            start_model_infer_time=time.time()
            eye_coord,right_eye,left_eye,face_image=landmark.predict(face_image)
            total_inf_time_facelandmark += (time.time() - start_model_infer_time)

            start_model_infer_time=time.time()
            frame,eye_center_coord=drawEyes(face_coords,eye_coord,frame)
            total_time_draw_eyes += (time.time() - start_model_infer_time)

            start_model_infer_time=time.time()
            pose_coord=headpose.predict(face_image)
            total_inf_time_headpose += (time.time() - start_model_infer_time)
            frame=writePoseAngles(frame,pose_coord)
            
            #Check for gaze when eyes are detected
            if( left_eye.size !=0 and right_eye.size !=0 ):
                start_model_infer_time=time.time()
                gaze_vect,mouse_coord=gaze.predict(left_eye,right_eye,pose_coord)
                total_ing_time_gazeest += (time.time() - start_model_infer_time)
                start_model_infer_time=time.time()
                frame=drawGazeAngles(frame,gaze_vect,eye_center_coord)
                total_time_gaze_vector += (time.time() - start_model_infer_time)
                if counter %5==0:
                     start_model_infer_time=time.time()
                     mousectrl.move(mouse_coord[0],mouse_coord[1])
                     total_time_move_ctrl+=(time.time() - start_model_infer_time)
            cv2.imshow('demo',frame)
            out_video.write(frame)
            if key==27:
                break
        except FailSafeException :
            log.info("mouse must have moved out of bounds")
            continue
        except Exception as e:
            log.error("Ran into Inference: ", e)
    log.info('Video stream ended')
    total_time=time.time()-start_inference_time
    total_inference_time=round(total_time, 1)
    fps=counter/total_inference_time


    log.info('total inference time for face detection = {:.2f}'.format(
         total_inf_time_facedetection))
    log.info('total inference time for face landmark = {:.2f}'.format(
         total_inf_time_facelandmark))
    log.info('total inference time for head pose = {:.2f}'.format(
         total_inf_time_headpose))
    log.info('total inference time for gaze estimate model = {:.2f}'.format(
         total_ing_time_gazeest))

    log.info('total time to draw eyes = {:.2f}'.format(
         total_time_draw_eyes))
    log.info('total time to draw gaze vectors = {:.2f}'.format(
         total_time_gaze_vector))
    log.info('total time taken by mouse controllers = {:.2f}'.format(
         total_time_move_ctrl))
    
    log.info('total inference time='+str(total_inference_time))
    log.info('FPS='+ str(fps))
    cv2.destroyAllWindows()
    feeder.close()
    


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-fm','--model_face', required=True)
    parser.add_argument('-fl','--model_landmark', required=True)
    parser.add_argument('-fh','--model_headpose', required=True)
    parser.add_argument('-fg','--model_gaze', required=True)
    parser.add_argument('-d','--device', default='CPU')
    parser.add_argument('-if','--input_file', default=None)
    parser.add_argument('-ift','--input_filetype', default='image')
    parser.add_argument('-th','--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)
