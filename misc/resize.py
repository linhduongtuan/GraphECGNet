import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob
from multiprocessing.pool import Pool


img_size = 512

def Resize(image):
    #image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # for gray-scale images
    image = Image.open(image).convert('L')
    #image = cv2.resize(image, (img_size, img_size))
    image = image.thumbnail((img_size, img_size))
    return image


def Resize_images(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        image = Resize(filename)
        # create the edge image and store it to consecutive filenames
        #cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', image)
        fnum = int(n)
        image.save(f"destdir/{fnum}.png")
        #filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

sourcedir = ('/home/linh/Downloads/data/ori_type/A')
destdir = ('/home/linh/Downloads/data/ori_resized_TYPE/A')
os.makedirs(destdir, exist_ok=True)
print("The new directory is created!")
#with Pool(28) as p:
#    p.map(Resize_images(sourcedir, destdir))
Resize_images(sourcedir, destdir)

