import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import random


#Load All Data
samples = []

with open('track1_normal/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('track1_normal_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
      
with open('track1_recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        

with open('track1_recovery_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


fig = plt.figure(figsize = (20,20))

for i in range(1,6):

	image_index = random.randint(0, len(samples))
	image = cv2.cvtColor(cv2.imread(samples[image_index][0]), cv2.COLOR_BGR2RGB)
	croppedImage = image[70:135, 0:320]
	angle = str(float(samples[image_index][3])*25)

	
	a = fig.add_subplot(2,5,i)
	a.set_title('Original Image - Steering Angle: \n{}'.format(angle))
	plt.axis('off')
	plt.imshow(image)
	a = fig.add_subplot(2,5,i+5)
	a.set_title('Cropped Image')
	plt.axis('off')
	plt.imshow(croppedImage)

plt.show()