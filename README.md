# The Detection of Dental Caries in Pediatric Bitewing Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 719 radiographs with 5 stages of dental caries
Healthy, Stage 2 (Inner Enamel), Stage 3 (Outer Dentin), Stage 4 (Middle Dentin) and Stage 5 (Inner Dentin)

## semi-enhanced Branch

This branch contains the implementation of YOLOv6 on first VOLOv6s and then YOLOv6s6 with processing to enhance the images using CLAHE

### 500 epochs YOLOv6s

9.2 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 122    | 117    | 0.31    | 0.261   | 0.278    | 0.216  | 0.0939     |

### 500 epochs YOLOv6s6

9.6 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 122    | 117    | 0.695   | 0.22    | 0.329    | 0.242  | 0.139      |
