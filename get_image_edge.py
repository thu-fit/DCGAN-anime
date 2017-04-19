import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def main()
    src_dir = "./data/anime"
    dest_dir = "./data/sketch"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    files = os.listdir(src_dir)
    l = len(files)
    for (i,fname) in enumerate(files):
        # here flags=0 means the return image is grayscale
        img = cv2.imread(os.path.join(src_dir, fname), flags=0)
        # detect edges
        edges = 255 - cv2.Canny(img, threshold1=160, threshold2=210)
        # save image
        dest_fname = os.path.join(dest_dir, fname)
        cv2.imwrite(dest_fname, edges)
        print("{}/{}".format(i,l))

if __name__ == "__main__":
    main()

