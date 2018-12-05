import cv2
import sklearn
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv2.imread('1.jpg', 0)

# Clean the image
img_dil = cv2.dilate(img, np.ones((7,7), np.uint8))
img_blur = cv2.medianBlur(img_dil, 21)
img_diff = 255 - cv2.absdiff(img, img_blur)
img_norm = img_diff.copy()
cv2.normalize(img_diff, img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
_, img_thr = cv2.threshold(img_norm, 230, 0, cv2.THRESH_TRUNC)
cv2.normalize(img_thr, img_thr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
img_final = 255 - img_thr

# plt.imshow(img_final, 'gray')
# plt.show()

# Get horizontal peaks
horizontal = np.sum(img_final, axis=1)
horizontal_peaks = []
horizontal_peak_start = -1

for i in range(horizontal.size):
    if horizontal[i] > 1100 and horizontal_peak_start == -1:
        horizontal_peak_start = i
    elif horizontal[i] < 300 and horizontal_peak_start != -1:
        horizontal_peaks.append({'start': horizontal_peak_start, 'end': i})
        horizontal_peak_start = -1

# To plot horizontal histograms

plt.plot(horizontal)
plt.show()

# To plot the horizontal slices

for i in range(len(horizontal_peaks)):
    hor_slice = img_final[horizontal_peaks[i]['start']: horizontal_peaks[i]['end']]
    plt.subplot(len(horizontal_peaks), 1, i+1)
    plt.imshow(hor_slice, 'gray')
plt.show()

# Get vertical peaks for each horizontal peaks

digits_bounds = []

for i in range(len(horizontal_peaks)):
    vertical = np.sum(img_final[horizontal_peaks[i]['start']: horizontal_peaks[i]['end']], axis = 0)
    plt.subplot(len(horizontal_peaks), 1, i+1)
    plt.plot(vertical)

    vertical_peak_start = -1
    for j in range(vertical.size):
        if vertical[j] > 150 and vertical_peak_start == -1:
            vertical_peak_start = j
        elif vertical[j] < 50 and vertical_peak_start != -1:
            digits_bounds.append({ \
                'y_start':horizontal_peaks[i]['start'], \
                'y_end': horizontal_peaks[i]['end'], \
                'x_start': vertical_peak_start, \
                'x_end' : j})
            vertical_peak_start = -1
plt.show()

digits_seg = []

for i in range(len(digits_bounds)):
    digit = img_final[digits_bounds[i]['y_start']: digits_bounds[i]['y_end'], digits_bounds[i]['x_start'] : digits_bounds[i]['x_end']]
    plt.subplot(2, int((len(digits_bounds) / 2)) + 1, i+1)
    plt.imshow(digit, 'gray')
    digits_seg.append(digit)
plt.show()

# vertical = np.sum(img_final, axis = 0)
# plt.plot(vertical)
# plt.show()

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# segmentation


# adaboost
