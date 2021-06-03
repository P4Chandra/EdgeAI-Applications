# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

#The process behind converting custom layers involves...

Identifying unsupported layers and
1) converting them to custom layers
2) or run the unsupported layers in their original framework
3) or use CPU extensions if the underlying hardware does not support certain layers.

#Some of the potential reasons for handling custom layers are...

One of the main reason being Model Optimizer might not support all layers, so one of the options is to convert unsupported layers to custom layers so that we then can run through Model Optimizer to generate IR(Intermediete Representation)

## Comparing Model Performance
ssd_mobilenet_v2_coco model was used. /home/workspace/public/ssd_mobilenet_v2_coco/

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...
Difficult to determine accuracy quantitatively. Pre-Run model detected people slightly better that post converted model. But there were instances in frames where it could not detect person. The model seems to be performing better on images then video streaming.

The size of the model pre- and post-conversion was...
pre-run model (frozen_inference_graph.pb):68,055KB
post-converted model (frozen_inference_graph.xml +frozen_inference_graph.bin) : 65,799KB

The inference time of the model pre- and post-conversion was...
Inference time wise pre-run model was 60-80 msec. Post inference is 40-50 msec.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1) To detect number of allowed people in restricted area for a certain duration of time
2) Count the people entering and leaving a shop or mall so crowd could be regulated
3) Attendance in factory or event or for surviellance
4) To count people crossing the roads so if high numbers,stall traffic for longer time to let people cross safely.

Each of these use cases would be useful because...
It ensures safety and security.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

1) Insuficient lighting can lead to false detection of people and objects
2) model accuracy matters as well if people are not detected across frames, the application will not perform reliably. If the model inference take time again application is rendered useless.
3) camera needs to take resonable resolution pictures for the model to detect efficiently.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v2_coco
  - Model Source :https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public : python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name ssd_mobilenet_v2_coco

  
  - I converted the model to an Intermediate Representation with the following arguments...
  python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

  Execution step: 
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - The model was insufficient for the app because...
  Did not detect person in few frames and hence recounted them as a new person
  - I tried to improve the model for the app by...
  - Increasing threshhold duration but that completely missed the person and did not count them.
  -Tried detecting person once every few frames,did not fix the double counting issue.In the  end although there were only 6 people in the video the Total Count showed 14-18.This model does better on images then videos.
  
  
- Model 2: ssd512
  - Model Source :https://github.com/weiliu89/caffe/tree/ssd : python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name ssd512
  
  - I converted the model to an Intermediate Representation with the following arguments...
  
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel --input_proto deploy.prototxt


  - The model was insufficient for the app because...
  Inference time was way too high around 2.5 seconds per frame
  - I tried to improve the model for the app by...
  Not sure what else could be done to improve this model

Execution step:
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m public/ssd512/models/VGGNet/VOC0712Plus/SSD_512x512/VGG_VOC0712Plus_SSD_512x512_iter_240000.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


- Model 3: person-detection-retail-0013
  - Intel Openvino pre-trained models
  
  - I converted the model to an Intermediate Representation with the following arguments...
  I downloaded the .bin and .xml 
  
  - The model was sufficient for the app because...
  it detected people in all the frames and hence total count matched to what is expected.
  
  - I tried to improve the model for the app by...
  I added few extra features like notofication if someone stays in the frame for too long
