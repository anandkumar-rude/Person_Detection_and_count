'''
This program performs 2 tasks:
1. Live Count Detection
2. Total count

** camera_source, dwell_time allowed and person_Count is taken from
json file. [config.json or default.json]
'''


#importing the required libraries
import cv2
import datetime
import time
import numpy as np
from tracker.centroidtracker import CentroidTracker
import threading
import json
import os
import sys 
import shapely
from shapely import geometry
from shapely.geometry import Point, Polygon, LineString, GeometryCollection



gpu = True

try:
    # check if config file exist
    check= os.path.exists('./config.json')
    if check == True:
        conf = json.load(open("./config.json")) #loads the values from config file
except:
    print("Configuration file missing!")
    pass

try:
    #get the model files for SSD model
    protopath = "./model/MobileNetSSD_deploy.prototxt.txt"
    modelpath = "./model/MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
except:
    print("Model Files missing!")
    pass

if gpu == True:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


#list of classes available
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#initiate the tracker object
tracker = CentroidTracker(maxDisappeared=15)


# non_max_suppression function to avoid the over lapping boxes
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


#function for geneing dwell_time alert and registering the IDs
# def dwell_time_alert(dwell_time,objectId, dwell_alert):
#     if dwell_time[objectId] > conf["person_duration"]:
#         if objectId not in dwell_alert:
#             print(f"ID:{objectId} time exceeded!!!")
#             # cv2.imwrite(f"time_alert id:{objectId}.jpg", frame)
#             dwell_alert.append(objectId)

#function for generating person_count alert and registering the IDs
def person_count_alert(objectId, perCount_alert, person_count, frame):
    if objectId not in perCount_alert:
        print(f"person count exceeded!! | Total Count: {person_count}")
        #path for saving the images
        log_path = "./person_count_alert"
        #create folder if does not exist
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t) #time stamp for saving image
        
        #save the image to specific path and append the object_id
        # cv2.imwrite(os.path.join(log_path, f"person Count exceeded{current_time}.jpg"), frame)
        perCount_alert.append(objectId)


#main driver function
def main():
    # #live_count = 0
    prev_count = 0
    person_exit = 0
    p_c =  0
    current_count =0
   # previous_count = 0
    # person_left = 0
    # #person_present = 0 
    # person_exit = 0
    # person_visited = 0

    # get the video from the source
    cap = cv2.VideoCapture(conf["vid_source"])

    #list for appending the Ids
    perCount_alert= []
    dwell_alert= []


    # lpc_count = 0
    object_id_list = []
    dtime = dict()
    dwell_time = dict()

    #checking if the feed is live from the source
    if (cap.isOpened() == False):
        print("Error opening video stream")
        sys.exit()



    while cap.isOpened():


        ret, frame = cap.read()
        frame = cv2.resize(frame, (640,510))
        box = frame[100:450, 100:500]

        (H, W) = box.shape[:2]

        blob = cv2.dnn.blobFromImage(box, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.65:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects) #convert the bounding box cordinates to np array
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.5)
        centroid_dict = dict()
        objects = tracker.update(rects)

        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
	    # get the centroid from the coordinates
            cX = int((x1 + x2) / 2.0) 
            cY = int((y1 + y2) / 2.0)
        #-------------------------------------------------------------------------------------
            cv2.rectangle(box, (x1, y1), (x2, y2), (0, 0, 255), 2)
            poly =  Polygon([(100,100), (100,450), (500,450), (500,100)])
            #poly
            point = geometry.Point(cX, cY)
            #polygon = geometry.Polygon(line)

            #print(poly.contains(point))
        #--------------------------------------------------------------------------------------
            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

            if objectId not in object_id_list:
                current_count+=1

        #---------------------------------------------------------------------------------------------            
                # if poly.contains(point) ==True:
                #     current_count+=1
                
        #             print("Inside the polygon Block ")  

        # # #------------------------------------------------------------------------------------- 
                object_id_list.append(objectId)

        # #             dtime[objectId] = datetime.datetime.now()
        # #            # dwell_time[objectId] = 0
        # #------------------------------------------------------------------------------------------
               
                

        #         if current_count == prev_count:
        #             print("Not change")
        #             pass
        #         if prev_count != current_count:
        #             print("changed!!!!!!!!!!!!")
        #             if prev_count == 0:
        #                 prev_count = current_count
        #                 pass

        #             if prev_count != current_count and current_count >= p_c:
        #                 prev_count = current_count

        #             if prev_count != current_count and current_count < p_c:
        #                 person_exit = prev_count - current_count
        #                 prev_count = current_count

        #             p_c = current_count

                # print("Current_count: ", current_count)
                # print("Previous_count: ", prev_count)
                # print("Exit: ", person_exit)


                   
                            

        #-------------------------------------------------------------------------------------------


            #if object_id is present in the object_id list
            #calculate the time difference between the current and old time in seconds
            # if objectId in object_id_list:
            #     curr_time = datetime.datetime.now()
            #     old_time = dtime[objectId]
            #     tedwell_time[objectId] += sec

            # check the dwell_time condition to send alert
            #thread2= threading.Thread(target=dwell_time_alert, args= [dwell_time,objectId, dwell_alert])
            #thread2.start()
            # dwell_time_alert(dwell_time,objectId, dwell_alert)

            #to print on the frame
           # text = "ID:{} | t:{}".format(objectId, int(dwell_time[objectId]))
          #  cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


        #for getting the person_count
        person_count = len(objects)

        # print on the frame
        lpc_txt = "Live Person Count: {}".format(person_count)
        cv2.putText(frame, lpc_txt, (1, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        tc_txt = "Total Count: {}".format(current_count)
        cv2.putText(frame, tc_txt, (1, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.rectangle(frame,(100,100),(500,450),(0, 0, 255), 2)
      
        # if person_count > conf["personCountExceed"]:
        #     thread3= threading.Thread(target= person_count_alert, args= [objectId, perCount_alert, person_count, frame])
        #     thread3.start()
            # person_count_alert(objectId, perCount_alert, person_count, frame)

        #display the feed
        cv2.imshow("Application", frame)
        key= cv2.waitKey(13)
        if key == ord('q'):
            break


    cv2.destroyAllWindows()

main_thread = main()
main_thread.start()
