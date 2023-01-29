from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu
import cv2
import glob


path = "C:/Users/aliel/Downloads/Dataset/Dataset/Image_training/normal/*.*"

img_number = 1
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 0)  #now, we can read each file since we have the full path

    if np.mean(img) == 0:
        print("Image is all black")
    elif np.mean(img) == 255:
        print("Image is all white")
        #continue
    else:
        
        thresholds = threshold_multiotsu(img, classes=3)
        # Digitize (segment) original image into multiple classes.
        #np.digitize assign values 0, 1, 2, 3, ... to pixels in each class.
        regions = np.digitize(img, bins=thresholds)
        plt.imshow(regions)
        segm1 = (regions == 0)
        segm2 = (regions == 1)
        segm3 = (regions == 2)
        
        all_segments = np.zeros((img.shape[0], img.shape[1], 3)) 
        all_segments[segm1] = (1,0,0)
        all_segments[segm2] = (0,1,0)
        all_segments[segm3] = (0,0,1)
        #all_segments_cleaned[segm4_closed] = (1,1,0)

        plt.imshow(all_segments)
        if img_number < 10:
            x= "00" + str(img_number)
        elif img_number < 100:
            x="0" + str(img_number)
        else:
                x= str(img_number)
                
        plt.imsave("C:/Users/aliel/Downloads/Dataset/Dataset/Image_training/normal/segmented/regions/"+ x +"regions"+".jpg", all_segments)
        plt.imsave("C:/Users/aliel/Downloads/Dataset/Dataset/Image_training/normal/segmented/plume/"+ x +"plume"+".jpg", segm1)
        #plt.imsave("C:/Users/aliel/Downloads/Dataset/Dataset/Image_training/normal/segmented/00"+str(img_number)+"plume"+".jpg", segm2)
        plt.imsave("C:/Users/aliel/Downloads/Dataset/Dataset/Image_training/normal/segmented/spatterpool/"+ x +"spatterpool"+".jpg", segm3)
        img_number +=1

    

        

