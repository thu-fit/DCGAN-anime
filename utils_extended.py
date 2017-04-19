import cv2
import numpy as np

def image2edge(imgs, threshold1=160, threshold2=210):
    '''
    edge detect for a batch of image

    input: 
        imgs: [batch_size, width, height, channel=3], dtype = float32
    output:
        result: [batch_size, width, height, 1], dtype = float32
    '''
    imgs = ((imgs + 1.) / 2. * 255.).astype(np.uint8)  # tranform to [0, 255], uint8
    result = np.zeros(imgs.shape[:3] + (1,), dtype=np.float32)
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # RGB to grascale
        edges = 255. - cv2.Canny(img, threshold1, threshold2)  # detect edge, and inverse color
        edges = ((edges / 255.) * 2. - 1.)  # inverse transform
        result[i] = edges.reshape([edges.shape[0], edges.shape[1], 1])
    return result

