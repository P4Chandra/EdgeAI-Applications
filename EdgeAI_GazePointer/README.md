# Computer Pointer Controller using OpenVino ToolKit and OpenVino Developer Bench.

This has been one among the most challenging  and interesting projects I have worked on. The project is about using your gaze to control the movement of mouse pointer on your computer screen. Due to absence of webcam on my PC,I used the demo video. However would definetly like to try out my head pose and gaze to control the pointer via webcam. 

Due to impending deadline I could not try out workbench to optimize my application.

## Project Set Up and Installation

### Install OpenVino Toolkit
The Project requires installation of openvino toolkit. the directions to install can be found in the link below:
 https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables

### Virtual Environment Creation
Create a virtual environment for the project by running following commands:
conda create -n edge_ai python=3.6
conda activate edge_ai (Activate the environment)
pip install -r requirements.txt  (Installs required software for this project-See Directory structure for location)

### Setup Openvino environment
C:\\"Program Files (x86)"\\Intel\\openvino_2021\\bin\setupvars.bat 

### Download openvino models for this project
python C:\\"Program Files (x86)"\\Intel\\openvino_2021\\deployment_tools\\tools\\model_downloader\\downloader.py --name <model_name>

Face detection:face-detection-adas-0001
Face Landmark detection:landmarks-regression-retail-0009
Head Pose Estimation:head-pose-estimation-adas-0001
Gaze Estimation : gaze-estimation-adas-0002

### Directory structure
-EdgeAI_GazePointer

--bin
---demo.mp4 (demo video used for testing application)
---image3.jpg

--intel
--face-detection-adas-0001
---FP16
----face-detection-adas-0001.bin
----face-detection-adas-0001.xml
--landmarks-regression-retail-0009
---FP16
----landmarks-regression-retail-0009.bin
----landmarks-regression-retail-0009.xml
--head-pose-estimation-adas-0001
---FP16
----head-pose-estimation-adas-0001.bin
----head-pose-estimation-adas-0001.xml
--gaze-estimation-adas-0002
---FP16
----gaze-estimation-adas-0002.bin
----gaze-estimation-adas-0002.xml

--src
---inputfeeder.py ( Captures video and generate frames)
---main.py (Main application that handles calls to models and processes their outputs)
---basemodel.py (Base Class definition for all 4 models - handles, initialization,loading model and model input processing)
---face_detection.py (Input: Video frame ::: Output: coordiantes of face and image of cropped face)
---facial_landmark_detection.py  ( Input: cropped face ::: Output: left eye and right eye coordiantes along with their cropped images)
---head_pose_estimation.py (Input: cropped face ::: Output: Pose of head :yaw,pitch and roll)
---gaze_estimation.py  (Input: head pose and images of eyes ::: Output: gaze direction )
---mouse_controller.py  (Input :Gaze direction coordinates )

--output_video.mp4 ( final demo video with mouse movement)


## Demo
Run the following command on Anaconda prompt in the same order as suggested inorder to launch application:
1) cd to project directory 

2)conda activate edge_ai

3) C:\\"Program Files (x86)"\\Intel\\openvino_2021\\bin\setupvars.bat 

4) python  src\main.py -fm intel\face-detection-adas-0001\FP16\face-detection-adas-0001 -fl intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009 -fh intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001 -fg intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002 -if bin\demo.mp4 -ift video



## Documentation
Lets break down the command line in 4) of Demo section. 
main.py is the application file that needs to be launched. The first argument (-fm) is the location of facedetection model, the second arg is location of (-fl) facial lanmark model, third arg is location of (-fh) head pose model and the 4th arg being location of gaze estimation model.
5th argument is the location of file . The last argument is type of file i.e. if it is 'video' or 'image'

usage: main.py [-h] -fm MODEL_FACE -fl MODEL_LANDMARK -fh MODEL_HEADPOSE -fg
               MODEL_GAZE [-d DEVICE] [-if INPUT_FILE] [-ift INPUT_FILETYPE]
               [-th THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -fm MODEL_FACE, --model_face MODEL_FACE
  -fl MODEL_LANDMARK, --model_landmark MODEL_LANDMARK
  -fh MODEL_HEADPOSE, --model_headpose MODEL_HEADPOSE
  -fg MODEL_GAZE, --model_gaze MODEL_GAZE
  -d DEVICE, --device DEVICE
  -if INPUT_FILE, --input_file INPUT_FILE
  -ift INPUT_FILETYPE, --input_filetype INPUT_FILETYPE
  -th THRESHOLD, --threshold THRESHOLD

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

### Running FP16 models:
INFO:root:Loaded Face Detector Model load time in 0.13 secs
INFO:root:Loaded Face Landmark Detector Model in 0.03 secs
INFO:root:Loaded HeadPose Model in 0.06 secs
INFO:root:Loaded Gaze Estimation Model in 0.07 secs
INFO:root:total inference time for face detection = 7.52
INFO:root:total inference time for face landmark = 0.74
INFO:root:total inference time for head pose = 0.75
INFO:root:total inference time for gaze estimate model = 0.76
INFO:root:total time to draw eyes = 0.02
INFO:root:total time to draw gaze vectors = 0.02
INFO:root:total time taken by mouse controllers = 160.06
INFO:root:total inference time=226.0
INFO:root:FPS=2.642762284196547

### Running FP32 models:
INFO:root:Loaded Face Detector Model in 0.12 secs
INFO:root:Loaded Face Landmark Detector Model in 0.03 secs
INFO:root:Loaded HeadPose Model in 0.04 secs
INFO:root:Loaded Gaze Estimation Model in 0.05 secs
INFO:root:total inference time for face detection = 7.46
INFO:root:total inference time for face landmark = 0.72
INFO:root:total inference time for head pose = 0.77
INFO:root:total inference time for gaze estimate model = 0.75
INFO:root:total time to draw eyes = 0.02
INFO:root:total time to draw gaze vectors = 0.02
INFO:root:total time taken by mouse controllers = 160.19
INFO:root:total inference time=226.0
INFO:root:FPS=2.6415929203539825

### Running INT8 models:
INFO:root:Loaded Face Detector Model in 0.22 secs
INFO:root:Loaded Face Landmark Detector Model in 0.05 secs
INFO:root:Loaded HeadPose Model in 0.06 secs
INFO:root:Loaded Gaze Estimation Model in 0.08 secs
INFO:root:total inference time for face detection = 5.82
INFO:root:total inference time for face landmark = 0.69
INFO:root:total inference time for head pose = 0.68
INFO:root:total inference time for gaze estimate model = 0.56
INFO:root:total time to draw eyes = 0.02
INFO:root:total time to draw gaze vectors = 0.03
INFO:root:total time taken by mouse controllers = 167.95
INFO:root:total inference time=229.6
INFO:root:FPS=2.6001742160278747

## Results
The inference time for individual models on INT8 is  lower than FP32 or FP16, model load time is a bit higher than FP16 or FP32.
However when I tested the model I found it lesser accurate than FP32 and FP16. FP16 and FP32 Load time are pretty close although FP16 load time is slightly better than FP32. I did not observe any performance or accuracy difference between FP16 vs FP32. However FP16 seems to have better inference time than FP32. 

Hence I would chose to use FP16
## Stand Out Suggestions
I am almost out of time and could not get to try any feature from standout suggestion.

### Async Inference
I have used ASYNC inference, inference time with ASYNC is lower and hence has better performance than SYNC. However this involves lot more computation and hence more power consumption. 

### Edge Cases
Absence of gaze when eyes closed: There were instances in video when gaze vectors could not be calculated as the eyes were closed. Hence I calcualted mouse meovement only when gaze estimation was available. I also calculated mouse movements once every 5 frames to make the movements smoother.

For multiple faces scenario in Face detection model. I calculated the area of rectange (boxed face). and selected the face with larger area The idea is larger area  means closer the face is to the camera. Hence mouse could be controlled by most significant face.
