# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 01:20:23 2022

@author: aliel
"""

#importing all the required modules
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
#reading the image on which bounding box is to be drawn using imread() function
image = cv.imread('C:/Users/aliel/Downloads/Dataset/Dataset/Image_training/normal/segmented/spatterpool/002spatterpool.jpg', 0)
#using selectROI() function to draw the bounding box around the required objects
#imagedraw = cv.selectROI(image)
#cropping the area of the image within the bounding box using imCrop() function
croppedimage = image[2:110, 0:200] #displaying the cropped image as the output on the screen
#croppedimage = image[int(imagedraw[1]):int(imagedraw[1]+imagedraw[3]), int(imagedraw[0]):int(imagedraw[0]+imagedraw[2])] 
#cv.imshow('Cropped_image',croppedimage)
#cv.waitKey(0)
#cv.destroyAllWindows()

threshold = threshold_otsu(croppedimage)
thresholded_img = croppedimage > threshold
plt.imshow(thresholded_img, cmap='gray')

#Remove edge touching regions
from skimage.segmentation import clear_border
edge_touching_removed = clear_border(thresholded_img)
plt.imshow(edge_touching_removed, cmap='gray')

#Label connected regions of an integer array using measure.label
#Labels each connected entity as one object
#Connectivity = Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
#If None, a full connectivity of input.ndim is used, number of dimensions of the image
#For 2D image it would be 2

label_image = measure.label(thresholded_img, connectivity=image.ndim)

plt.imshow(label_image)
#Return an RGB image where color-coded labels are painted over the image.
#Using label2rgb

image_label_overlay = label2rgb(label_image, image=croppedimage)
plt.imshow(image_label_overlay)

plt.imsave("labeled_cast_iron.jpg", image_label_overlay) 

#################################################
#Calculate properties
#Using regionprops or regionprops_table
all_props=measure.regionprops( croppedimage)
#Can print various parameters for all objects
for prop in all_props:
    print('Label: {} Area: {}'.format(prop.label, prop.area))

#Compute image properties and return them as a pandas-compatible table.
#Available regionprops: area, bbox, centroid, convex_area, coords, eccentricity,
# equivalent diameter, euler number, label, intensity image, major axis length, 
#max intensity, mean intensity, moments, orientation, perimeter, solidity, and many more

props = measure.regionprops_table(label_image, croppedimage, properties=['label','area','equivalent_diameter','solidity' , "centroid"] )

import pandas as pd
df = pd.DataFrame(props)
print(df.head())

#To delete small regions...
df = df[df['area'] > 20]
print(df.head())

#######################################################
#Convert to micron scale
#df['area_sq_microns'] = df['area'] * (scale**2)
#df['equivalent_diameter_microns'] = df['equivalent_diameter'] * (scale)
#print(df.head())

df.to_csv('cast_iron_measurements001.csv')
