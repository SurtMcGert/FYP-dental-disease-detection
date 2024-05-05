# The Detection of Dental Caries in Panoramic Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries provided by DENTEX
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## self-attention Branch

This branch contains the implementation of Self Attention on the YOLOv6 Model

### 500 epochs YOLOv6s + Self Attention imp1

13.27 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.874   | 0.83    | 0.851    | 0.866  | 0.627      |

### 500 epochs YOLOv6s + Self Attention imp2

12.20
| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all | 141 | 732 | 0.943 | 0.785 | 0.854 | 0.854 | 0.629 |
