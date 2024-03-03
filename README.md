# The Detection of Dental Caries in Pediatric Bitewing Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries
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
| all   | 122    | 117    | 0.695   | 0.22    | 0.329    | 0.242  | 0.139      |

### 500 epochs YOLOv6s6 SGD Optimized

10.3 hours to get hyperparams, 9.6 hours to train

best mAP: 0.013285234570503235
the best hyperparameters: tensor([1.0000e-01, 1.0000e-03, 9.0000e-01, 1.0000e-04])

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 122    | 117    | 0.569   | 0.235   | 0.332    | 0.239  | 0.147      |

### 500 epochs YOLOv6s6 Adam Optimized

12 hours to get hyperparams, 29.35 hours to train

best mAP: 0.03678618744015694
the best hyperparameters: tensor([0.0304, 0.0676, 0.8504, 0.0010])
didnt work so changed

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 122    | 117    | 0.336   | 0.247   | 0.278    | 0.2    | 0.11       |
