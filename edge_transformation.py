import numpy as np
import cv2
import os
import time
import glob
from multiprocessing.pool import Pool

img_size=224
blur_ksize=3 #1 or 3 or 5 or 7
#applying filter on a single image
def Prewitt_v1(filename, filter):
    print("reading file---> " + str(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #for gray-scale images
    img = cv2.resize(img,(img_size, img_size))
    #comment out the above line if there is memory issue i.e. need to resize all images to smaller dim
    h, w = img.shape # height and width of images
    print("shape: height " + str(h)+" x width " + str(w) + "\n")

    # define filters
    horizontal = filter
    vertical   = np.transpose(filter)

    # define images with 0s
    newhorizontalImage = np.zeros((h, w))
    newverticalImage   = np.zeros((h, w))
    newgradientImage   = np.zeros((h, w))

    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                             (horizontal[0, 1] * img[i - 1, j]) + \
                             (horizontal[0, 2] * img[i - 1, j + 1]) + \
                             (horizontal[1, 0] * img[i, j - 1]) + \
                             (horizontal[1, 1] * img[i, j]) + \
                             (horizontal[1, 2] * img[i, j + 1]) + \
                             (horizontal[2, 0] * img[i + 1, j - 1]) + \
                             (horizontal[2, 1] * img[i + 1, j]) + \
                             (horizontal[2, 2] * img[i + 1, j + 1])

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                           (vertical[0, 1] * img[i - 1, j]) + \
                           (vertical[0, 2] * img[i - 1, j + 1]) + \
                           (vertical[1, 0] * img[i, j - 1]) + \
                           (vertical[1, 1] * img[i, j]) + \
                           (vertical[1, 2] * img[i, j + 1]) + \
                           (vertical[2, 0] * img[i + 1, j - 1]) + \
                           (vertical[2, 1] * img[i + 1, j]) + \
                           (vertical[2, 2] * img[i + 1, j + 1])

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

    return newgradientImage


def Prewitt_v2(image):
       print("reading file---> " + str(image))
       image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #for gray-scale images
       image = cv2.resize(image, (img_size, img_size))
      # Prewitt operator
       kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
       kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
       x = cv2.filter2D(image, cv2.CV_16S, kernelx)
       y = cv2.filter2D(image, cv2.CV_16S, kernely)

       # Turn uint8, image fusion
       absX = cv2.convertScaleAbs(x)
       absY = cv2.convertScaleAbs(y)
       Prewitt_v2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
       return Prewitt_v2

def converter_Canny(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        image    = cv2.imread(filename)
        image    = cv2.resize(image, (img_size, img_size))
        imagemat = cv2.Canny(image, 100, 200)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

def Sobel_function(image_path, blur_ksize=5, sobel_ksize=1, skipping_threshold=30):
    """
    image_path: link to image
    blur_ksize: kernel size parameter for Gaussian blurry
    sobel_ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7
    skipping_thresholdL ignore weakly edge
    """
    # read image
    image        = cv2.imread(image_path)

    # resize image
    image        = cv2.resize(image, (img_size, img_size))

    # Convert BGR to GrayScale
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Sobel algorithm use cv2.CV_64F
    sobel_x64f    = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobelx64f = np.absolute(sobel_x64f)
    img_sobelx    = np.uint8(abs_sobelx64f)

    sobel_y64f    = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobely    = np.absolute(sobel_y64f)
    img_sobely    = np.uint8(abs_sobely)

    # Calculate magnitude/gradient
    img_sobel     = (img_sobelx + img_sobely) / 2

    # ignore weakly pixel
    for i in range(img_sobel.shape[0]):
        for j in range(img_sobel.shape[1]):
            if img_sobel[i][j] < skipping_threshold:
                img_sobel[i][j] = 0
            else:
                img_sobel[i][j] = 225
    return img_sobel

def converter_Sobel_v1(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        image    = Sobel_function(image_path=filename, blur_ksize=7, sobel_ksize=1, skipping_threshold=30)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', image)
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


def converter_Sobel_v2(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        # Read the original image
        image    = cv2.imread(filename,flags=0)
        image    = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
        image    = cv2.resize(image,(img_size, img_size)) 
        # Blur the image for better edge detection
        #image = cv2.GaussianBlur(image, (3,3), SigmaX=0, SigmaY=0)
        # Sobel Edge Detection
        #sobelx   = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        #sobely   = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy  = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', sobelxy) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

#function for creating all edge-images of a directory
def converter_Prewitt_v1(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        #applying Prewitt filter
        #for appyling any other filter change filter value accordingly i.e. the 2nd args for Prewitt filter version 1
        imagemat = Prewitt_v1(filename, np.array([[-1,0,1], [-1,0,1], [-1,0,1]]))
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


#function for creating all edge-images of a directory
def converter_Prewitt_v2(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        #applying Prewitt filter
        #for appyling any other filter change filter value accordingly i.e. the 2nd args for Prewitt filter version 2
        imagemat = Prewitt_v2(filename)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")



start = time.time()


#sourcedir = '/Users/mac/Downloads/TUH_EEG/Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Prewitt_v1/Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Prewitt_v2/Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Sobel_v2/Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Canny/Epilepsy'



sourcedir = '/Users/mac/Downloads/TUH_EEG/No_Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Prewitt_v1/No_Epilepsy'
destdir   = '/Users/mac/Downloads/TUH_EEG/Prewitt_v2/No_Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Sobel_v2/No_Epilepsy'
#destdir   = '/Users/mac/Downloads/TUH_EEG/Canny/No_Epilepsy'


os.makedirs(destdir, exist_ok=False)
print("The new directory is created!")
with Pool(28) as p:
    #p.map(converter_Canny(sourcedir, destdir))
    #p.map(converter_Sobel_v1(sourcedir, destdir))
    #p.map(converter_Sobel_v2(sourcedir, destdir))
    p.map(converter_Prewitt_v1(sourcedir, destdir))
    #p.map(converter_Prewitt_v2(sourcedir, destdir))

end = time.time()
time_to_transform = (end - start)/60
print("Total time (min) for transforming edege :", time_to_transform)
print("=======End transforming edege process here======")
