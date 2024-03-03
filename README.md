# The Detection of Dental Caries in Panoramic Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries provided by DENTEX
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## main Branch

This branch contains a basic implementation of YOLOv6 trained on the dataset with no pre-processing

### 500 epochs

5.5 hours
| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all | 71 | 433 | 0.45 | 0.524 | 0.484 | 0.461 | 0.263 |
