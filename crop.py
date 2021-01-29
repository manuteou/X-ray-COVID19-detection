import cv2
import numpy as np
from pathlib import Path

class crop:

    def crop(self,filename):
        seg = 'static/images/zoom.png' #load the segmented image template for crop
        
        img_rgb = cv2.imread("C:/Users/froge/Exo_jedha/PROJET/Site/static/Analyse/" + filename)
        img_rgb = cv2.resize(img_rgb,(250,250))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # we transfrom rgb image in grey image
        template = cv2.imread(seg,0)                        # for the calcul of coordonate

        h,w = template.shape[::]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
        
        _, _, min_loc, _ = cv2.minMaxLoc(res)
        return img_rgb[min_loc[0]:min_loc[0] + w,min_loc[1]:min_loc[1] + h]
