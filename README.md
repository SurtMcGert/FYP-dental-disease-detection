# The Detection of Dental Caries in Panoramic Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries provided by DENTEX
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## semi-enhanced Branch

This branch contains the implementation of YOLOv6 on first VOLOv6s and then YOLOv6s6 with processing to enhance the images using CLAHE

### 500 epochs YOLOv6s

10.8 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.925   | 0.771   | 0.839    | 0.867  | 0.629      |

### 500 epochs YOLOv6s6

9.6 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.914   | 0.824   | 0.866    | 0.873  | 0.635      |

### 500 epochs YOLOv6s6 SGD Optimized

0.13 hours to make initial weights 11.62 hours to get hyperparams, 12.10 hours to train

best mAP: 0.11704626679420471
the best hyperparameters: tensor([1.0000e-01, 1.0000e-03, 8.8008e-01, 1.0000e-04])

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.924   | 0.805   | 0.859    | 0.857  | 0.648      |

### 500 epochs YOLOv6s6 Adam Optimized

0.13 hours to make initial weights 12.00 hours to get hyperparams, 15.14 hours to train

best mAP: 0.3040043115615845
the best hyperparameters: tensor([4.7758e-02, 1.0000e-03, 9.0000e-01, 1.0000e-04])

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.774   | 0.698   | 0.732    | 0.755  | 0.5        |
