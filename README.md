# The Detection of Dental Caries in Panoramic Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries provided by DENTEX
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## self-attention Branch

This branch contains the implementation of Self Attention on the YOLOv6 Model

### 10 epochs YOLOv6s + Self Attention imp1 + design 1 pre-processing and pretrained weights

0.89 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.9     | 0.839   | 0.868    | 0.88   | 0.645      |
