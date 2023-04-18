import cv2
import os, glob
import numpy as np
#image = cv2.imread('/Users/mac/Downloads/MIT_ECG/dat/APC/APC_100182923820.png', cv2.IMREAD_GRAYSCALE)
#inverted = np.invert(image)

#cv2.imwrite('/Users/mac/Downloads/MIT_ECG/inverted.jpg', inverted)

def Invert(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  
    image = np.invert(image)
    return image

def Invert_images(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        image = Invert(filename)
        # create the edge image and store it to consecutive filenames
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', image)
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

sourcedir = ('/Users/mac/Downloads/MIT_ECG/dat/VFW')
destdir = ('/Users/mac/Downloads/MIT_ECG/d/VFW')
os.makedirs(destdir, exist_ok=True)
print("The new directory is created!")
#with Pool(28) as p:
#    p.map(Resize_images(sourcedir, destdir))
Invert_images(sourcedir, destdir)