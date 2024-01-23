# The Detection of Dental Caries in Pediatric Bitewing Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 719 radiographs with 5 stages of dental caries
Healthy, Stage 2 (Inner Enamel), Stage 3 (Outer Dentin), Stage 4 (Middle Dentin) and Stage 5 (Inner Dentin)

## main Branch

This branch contains a basic implementation of YOLOv6 trained on the dataset with no pre-processing

### 500 epochs

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 72     | 112    | 0.375   | 0.189   | 0.209    | 0.153  | 0.054      |

## Increased-Dataset Branch

This branch contains the same implementation as main, except the dataset has been increased in size to 1438 through a process of rotating and flipping images
