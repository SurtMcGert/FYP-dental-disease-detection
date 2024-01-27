# The Detection of Dental Caries in Pediatric Bitewing Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 719 radiographs with 5 stages of dental caries
Healthy, Stage 2 (Inner Enamel), Stage 3 (Outer Dentin), Stage 4 (Middle Dentin) and Stage 5 (Inner Dentin)

## main Branch

This branch contains a basic implementation of YOLOv6 trained on the dataset with no pre-processing

### 500 epochs

5.5 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 72     | 112    | 0.375   | 0.189   | 0.209    | 0.153  | 0.054      |

## increased-dataset Branch

This branch contains the same implementation as main, except the dataset has been increased in size to 1438 images through a process of rotating and flipping images

### 500 epochs

9.2 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 122    | 117    | 0.27    | 0.301   | 0.27     | 0.201  | 0.0723     |

## enhanced-dataset Branch

This branch adds image processing to each image to create an enhanced image with an extra edge detection layer

### 500 epochs

30 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 122    | 117    | 0.257   | 0.23    | 0.233    | 0.177  | 0.0612     |

## semi-enhanced-dataset branch

This branch adds image enhancement but no extra edge detect layer

### 500 epochs
