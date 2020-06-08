import cv2
import numpy as np
import os
class Lesion:
    """
    This is the main class and contains methods to perform various steps.

    It essentially performs following steps:
        1. Read a lesion image and validate the image
        2. Preprocess the image by applying color transformation and filtering
        3. Segment the lesion from the image using active contour model
        4. Extract features (A, B, C, D) from the lesion
        5. Classify the lesion based on the features extracted using an SVM
           classifier and output the result.
    """
    def __init__(self, file_path, iterations=3):
        """
        Initiate the program by reading the lesion image from the file_path.

        :param file_path:
        :param iterations:
        """
        self.file_path = file_path
        self.base_file, _ = os.path.splitext(file_path)
        self.original_image = cv2.imread(file_path)
        self.image = None
        self.segmented_img = None
        self.hsv_image = None
        self.contour_binary = None
        self.contour_image = None
        self.contour_mask = None
        self.warp_img_segmented = None
        self.color_contour = None
        self.asymmetry_vertical = None
        self.asymmetry_horizontal = None
        self.results = None
        self.value_threshold = 150
        self.iterations = int(iterations)

        # dataset related params (PH2)
        self.real_diamter_pixels_mm = 72
        self.hsv_colors = {
            'Blue Gray': [np.array([15, 0, 0]),
                          np.array([179, 255, self.value_threshold]),
                          (0, 153, 0), 'BG'],  # Green
            'White': [np.array([0, 0, 145]),
                      np.array([15, 80, self.value_threshold]),
                      (255, 255, 0), 'W'],  # Cyan
            'Light Brown': [np.array([0, 80, self.value_threshold + 3]),
                            np.array([15, 255, 255]), (0, 255, 255), 'LB'],
            # Yellow
            'Dark Brown': [np.array([0, 80, 0]),
                           np.array([15, 255, self.value_threshold - 3]),
                           (0, 0, 204), 'DB'],  # Red
            'Black': [np.array([0, 0, 0]), np.array([15, 140, 90]),
                      (0, 0, 0), 'B'],  # Black
        }
        self.iter_colors = [
            [50, (0, 0, 255)],
            [100, (0, 153, 0)],
            [200, (255, 255, 0)],
            [400, (255, 0, 0)]
        ]
        self.borders = 2
        self.isImageValid = False
        self.contour = None
        self.max_area_pos = None
        self.contour_area = None
        self.feature_set = []
        self.performance_metric = []
        self.xmlfile = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "opencv_svm.xml")
        self.cols = ['A1', 'A2', 'B', 'C', 'A_B', 'A_BG', 'A_DB', 'A_LB',
                     'A_W', 'D1', 'D2']

        # Active contour params
        self.iter_list = [75, 25]
        self.gaussian_list = [7, 1.0]
        self.energy_list = [2, 1, 1, 1, 1]
        self.init_width = 0.65
        self.init_height = 0.65
        self.shape = 0  # 0 - ellipse, 1 - rectangle

    def preprocess(self):
        """
        Validate the image and preprocess the image by applying smoothing
        filter and color transformation.

        :return: True if succeeded else None
        """
        try:
            if self.original_image is None:
                self.isImageValid = False
                return
            if self.original_image.shape[2] != 3:
                self.isImageValid = False
                return

            # morphological closing
            self.image = self.original_image.copy()
            # blur image
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
            # Applying CLAHE to resolve uneven illumination
            # hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            # hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            # self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # kernel = np.ones((11, 11), np.uint8)
            # for i in range(self.image.shape[-1]):
            #     self.image[:, :, i] = cv2.morphologyEx(
            #         self.image[:, :, i],
            #         cv2.MORPH_CLOSE, kernel)
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            self.contour_image = np.copy(self.original_image)
            self.isImageValid = True
            # Mednode dataset related params
            # if "mednode" in self.file_path:
            #     self.real_diamter_pixels_mm = (104 * 7360) // (
            #                 330 * 24)  # pixels/mm
            if self.iterations in range(3):
                temp = self.iter_colors[self.iterations]
                self.iter_colors.remove(self.iter_colors[self.iterations])
                self.iter_colors.append(temp)
            return True
        except:
            print("error")
            self.isImageValid = False
            return


img = cv2.imread('C:/Users/Leo/PycharmProjects/CONTORNO/lunar2.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lesion = Lesion('C:/Users/Leo/PycharmProjects/CONTORNO/lunar2.jpg')
if lesion.preprocess():
    print("PAQUINJICO")
else:
    print("EL PECOKPO")
cv2.imshow('FOTITO', lesion.hsv_image)
cv2.waitKey(0)
retval, thresh = cv2.threshold(lesion.hsv_image, 127, 255, 0)
#retval, thresh = cv2.threshold(gray_img, 127, 255, 0)
img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, img_contours, -1, (0, 255, 0))

cv2.drawContours(lesion.hsv_image, img_contours, -1, (0, 255, 0))
cv2.imshow('Image Contours', lesion.hsv_image)
cv2.waitKey(0)



"""
import cv2

img = cv2.imread('py1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, thresh = cv2.threshold(gray_img, 127, 255, 0)

img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, img_contours, -1, (0, 255, 0))
cv2.imshow('Image Contours', img)
cv2.waitKey(0)
"""
