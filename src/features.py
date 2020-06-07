# -*- coding: utf-8 -*-
"""
Methods to extract features (A, B, D) of a lesion.

Created on Tue Jun 06 11:29:16 2017

@author: ukalwa
"""
# third-party imports
import cv2
import numpy as np


def extract(image, mask, contour):
    """
    This method extracts Asymmetry, Border, and Diamter features along with
    lesion area, centroid, and perimeter. Performs affine transformation

    :param image: 3-d numpy array of an RGB image
    :param mask: binary image of the lesion mask image
    :param contour: list of contour points of the lesion
    :return: a list of all the features along with area, centroid,
    perimeter of the lesion, and transformed image
    """
    moments = cv2.moments(contour)
    #            contour_area = moments['m00']
    # (mH, mW) = mask.shape[:2]
    contour_area = cv2.countNonZero(mask)
    try:
        contour_centroid = [int(moments['m10'] / moments['m00']),
                            int(moments['m01'] / moments['m00'])]
        contour_perimeter = cv2.arcLength(contour, True)

        # Get max and min diameter
        rect = cv2.fitEllipse(contour)
        (x, y) = rect[0]
        (w, h) = rect[1]
        angle = rect[2]

        if w < h:
            if angle < 90:
                angle -= 90
            else:
                angle += 90
        rows, cols = mask.shape
        rot = cv2.getRotationMatrix2D((x, y), angle, 1)
        cos = np.abs(rot[0, 0])
        sin = np.abs(rot[0, 1])
        nW = int((rows * sin) + (cols * cos))
        nH = int((rows * cos) + (cols * sin))

        rot[0, 2] += (nW / 2) - cols / 2
        rot[1, 2] += (nH / 2) - rows / 2

        warp_mask = cv2.warpAffine(mask, rot, (nH, nW))
        warp_img = cv2.warpAffine(image, rot, (nH, nW))
        warp_img_segmented = cv2.bitwise_and(warp_img, warp_img,
                                             mask=warp_mask)

        im_mask, cnts, hierarchy = cv2.findContours(warp_mask, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in cnts]
        contour = cnts[np.argmax(areas)]
        xx, yy, nW, nH = cv2.boundingRect(contour)
        #        cv2.rectangle(warp_mask,(xx,yy),(xx+w,yy+h),(255,255,255),2)
        warp_mask = warp_mask[yy:yy + nH, xx:xx + nW]

        # get horizontal asymmetry
        flipContourHorizontal = cv2.flip(warp_mask, 1)
        flipContourVertical = cv2.flip(warp_mask, 0)

        diff_horizontal = cv2.compare(warp_mask, flipContourHorizontal,
                                      cv2.CV_8UC1)
        diff_vertical = cv2.compare(warp_mask, flipContourVertical,
                                    cv2.CV_8UC1)

        diff_horizontal = cv2.bitwise_not(diff_horizontal)
        diff_vertical = cv2.bitwise_not(diff_vertical)

        h_asym = cv2.countNonZero(diff_horizontal)
        v_asym = cv2.countNonZero(diff_vertical)

        return [{'area': int(contour_area), 'centroid': contour_centroid,
                 'perimeter': int(contour_perimeter),
                 'B': round(
                     (contour_perimeter ** 2) / (4 * np.pi * contour_area), 2),
                 'D1': max([nW, nH]), 'D2': min([nW, nH]),  # Normalize params
                 'A1': round(float(h_asym) / contour_area, 2),
                 'A2': round(float(v_asym) / contour_area, 2)},
                cv2.bitwise_not(diff_horizontal),
                cv2.bitwise_not(diff_vertical),
                warp_img_segmented]
    except:
        return {}
