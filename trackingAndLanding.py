import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy as np
import time
import tensorflow as tf
import os
import six.moves.urllib as urllib
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import math
from colorama import Fore,Back,Style

#Global Variables
F=1002.907 #Focal length
W=0.22 #Width of object in meters
maxBoxes=1 #max boxes to draw
minScoreThreshold=0.7 #minimum score to draw bounding box
drone = tellopy.Tello()


def handler(event,sender,data, **args):
	drone = sender
	if event is drone.EVENT_FLIGHT_DATA:
		print(data)
		print()

NUM_CLASSES=1
PATH_TO_MODEL = "PATH_TO_INFERENCE_GRAPH"
PATH_TO_LABELS = "PATH_TO_LABEL_MAP"
detection_graph=tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_MODEL,'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def,name ='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
	label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)	

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)

def printBoxAttributes(xmin,xmax,ymin,ymax):
	print("Box attributes")
	print("xmin: "+str(xmin)+"\txmax: "+str(xmax))
	print("ymin: "+str(ymin)+"\tymax: "+str(ymax))


def calculateDistance(boxWidth):
	'''
		For calculating the horizontal distance of the platform from the drone

		returns the distance in meters
	'''
	dist=round(W*F/boxWidth,2)
	print("Clalculated Distance = " + str(dist) + "m")
	return dist


def calculateRotationAngle(xmin,boxWidth,imageWidth):
	'''
		For calculating the angle of the platform from the drone

			-theta>0: drone rotates clockwise, platform is positioned right from the drone
			-theta<0: drone rotates counter clockwise, platform is positioned left from the drone
		
		returns the angle in degrees
	'''
	theta = ((82.6*xmin + 41.3*boxWidth)/imageWidth) - 41.3
	print("Drone rotating at "+str(round(theta,2))+"degrees")
	return round(theta,None)

def calculateVelocityOfPlatform(distance, lastKnownDistance,velocityTimer):
	'''
		For calculating the platform velocity.

		returns the velocity in meters per second
	'''
	velocity = (distance-lastKnownDistance)/(time.time()-velocityTimer)
	print(Fore.GREEN+"Platform velocity: "+str(velocity)+"m/s"+Fore.RESET)
	return round(velocity,2)


def rotationDrone(xmin,boxWidth,imageWidth):
	'''
		For controlling the rotation of the drone depending on the angle of the platform from the drone.
	'''
	theta=calculateRotationAngle(xmin,boxWidth,imageWidth)
	if theta>0:
		drone.clockwise(theta+10)
	else:
		drone.counter_clockwise(theta*(-1)+10)


def horizontalMovementDrone(distance,platformVelocity,firstCalculation,droneVelocity):
	'''
		For controling the speed of the drone depending on the velocity and the distance of the platform.

			-the further away the drone is from the platform, the faster it goes.
			-the drone slows down when it gets close to the platform for smoother landing
	'''
	if not firstCalculation:
		if distance>1.5 and (platformVelocity>0 or platformVelocity<-0.5):
			droneVelocity=droneVelocity+(platformVelocity*100)
		elif distance>1.0 and (platformVelocity>0 or platformVelocity<-0.2):
			droneVelocity=droneVelocity+(platformVelocity*100)
		elif distance>0.4 and (platformVelocity>0 or platformVelocity<-0.05):
			droneVelocity=droneVelocity+(platformVelocity*100)
		if droneVelocity>=0 and droneVelocity<=100:
			drone.forward(round(droneVelocity,None))
		elif droneVelocity>100:
			drone.forward(100)
			droneVelocity=100
		else:
			droneVelocity=0
			drone.forward(0)
	elif firstCalculation:
		droneVelocity=0
	return False, droneVelocity


def verticalMovementDrone(ymax, imageHeight, distance):
	'''
		For controling the downwards movement of the drone depending on the position of the platform.
	'''
	if distance>0.4:
		if (ymax>imageHeight-10):
			drone.down(40)
		elif (ymax > imageHeight-50):
			drone.down(30)
		elif (ymax > imageHeight-100):
			drone.down(20)
		elif (ymax > imageHeight-200):
			drone.down(10)
		else:
			drone.down(0)
	else:
		drone.land()

def scanningForPlatformDrone(lossCounter,droneVelocity):
	'''
		The drone stops moving after certain frames of not detecting anything to avoid crashing. 
	'''
	lossCounter=lossCounter+1
	if (lossCounter>100):
		drone.forward(0)
		drone.counter_clockwise(0)
		drone.clockwise(0)
		drone.down(0)
		droneVelocity=0
	return lossCounter,droneVelocity
	

	

def main():
	scale_percent = 100 # percent of original size
	imageWidth = int(960 * scale_percent / 100)
	imageHeight = int(720 * scale_percent / 100)
	dim = (imageWidth, imageHeight)
	firstVelocityCalculation=True
	platformVelocity=0
	velocityTimer=-1
	lastKnownDistance=0
	droneVelocity=0
	lossCounter=0
	previousBoxWidth=0
	try:
		drone.connect()
		drone.wait_for_connection(60.0)
		container = av.open(drone.get_video_stream())
		frame_skip = 300
		t0= time.time() #starting timer
		isOnAir=True
		hasTakeOff=False
		waitingSeconds=90 #seconds before emergency landing
		with detection_graph.as_default():
			with tf.compat.v1.Session(graph=detection_graph) as sess:
				for frame in container.decode(video=0):
					if 0 < frame_skip:
						frame_skip = frame_skip - 1
						continue
					start_time = time.time()
					image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
					if not hasTakeOff and time.time()-t0>15: #wait 15 before takeoff so the video would start
						hasTakeOff=True
						print(Fore.YELLOW+"TAKE OFF"+Fore.RESET)
						drone.takeoff()
						continue
					if (time.time()-t0) > waitingSeconds and isOnAir:
						print( "Emergency Landing")
						drone.land()
						isOnAir=False
					image_np_expanded = np.expand_dims(image, axis=0)
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					scores = detection_graph.get_tensor_by_name('detection_scores:0')
					classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
					#Detection
					(boxes, scores, classes, num_detections) = sess.run(
						[boxes, scores, classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})
					# Visualization of the results of a detection.
					vis_util.visualize_boxes_and_labels_on_image_array(
						image,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						max_boxes_to_draw=maxBoxes,
						min_score_thresh=minScoreThreshold,
						use_normalized_coordinates=True,
						line_thickness=4)
					if (scores[0]<minScoreThreshold).all():
						lossCounter,droneVelocity=scanningForPlatformDrone(lossCounter,droneVelocity)
					else:
						ymin = int((boxes[0][0][0]*imageHeight))
						xmin = int((boxes[0][0][1]*imageWidth))
						ymax = int((boxes[0][0][2]*imageHeight))
						xmax = int((boxes[0][0][3]*imageWidth))
						printBoxAttributes(xmin,xmax,ymin,ymax)
						boxWidth=xmax-xmin
						boxHeight=ymax-ymax
						distance= calculateDistance(boxWidth) #distance from the object
						if (time.time()-velocityTimer>1 or velocityTimer==-1):
							platformVelocity = calculateVelocityOfPlatform(distance,lastKnownDistance,velocityTimer)
							velocityTimer=time.time()
							lastKnownDistance=distance
						firstVelocityCalculation, droneVelocity = horizontalMovementDrone(distance,platformVelocity,firstVelocityCalculation, droneVelocity)
						rotationDrone(xmin,boxWidth,imageWidth) #rotating drone
						verticalMovementDrone(ymax,imageHeight,distance)


					#Display output
					cv2.imshow('object detection', image)
					cv2.waitKey(1)
					if frame.time_base < 1.0 / 60:
						time_base = 1.0 / 60
					else:
						time_base = frame.time_base
					#frame_skip = 10
					frame_skip = int((time.time() - start_time)/time_base)
	except Exception as ex:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		traceback.print_exception(exc_type, exc_value, exc_traceback)
		print(ex)
	finally:
		drone.land()
		time.sleep(2)
		drone.quit()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
