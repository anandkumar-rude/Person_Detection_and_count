# Description:

The usecase is built on jetson Nano(4GB dev kit)<br>
The usecase inference was executed and tested on jetson Nano.<br>

### This usecase is for generating alert: <br>
    1. when person is detected in a given area.
    2. if count of person increases beyond allowed number.


**SSD Model** is used for detecting people.<br>

## **Project Structure:**<br>

### Person_Detection_and_count <br>
    --- 1. Person_Detection_and_count.py
    --- 2. config.json 
    --- 3. model
        ------ MobileNetSSD_deploy.caffemodel
        ------ MobileNetSSD_deploy.prototxt.txt
    --- 4. tracker
        ------ centroidtracker.py
    
## **Parameters** and **values** for generating alert are defined in **config.json**<br>
### config.json containes the following parameters:<br>
  
    1. vid_source (video source[videofile, rtsp, camera])


**Note: To use Opencv with CUDA and use opencv dnn module install Opencv from source.**

## Steps to run the the script:
1. Define the following in the config.json file:<br>
    1. video source<br>

2. Open the terminal in the location where **Person_Detection_and_count.py** is present.<br>

3. run the following command:
> $ python3 Person_Detection_and_count.py
