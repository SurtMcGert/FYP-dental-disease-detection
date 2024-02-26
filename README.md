# The Detection of Dental Caries in Panoramic Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 classes
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted tooth

## Increased-Dataset Branch

This branch contains the same implementation as main, except the dataset has been increased to double its size through a process of rotating and flipping images

# 500 epochs

10.823

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.947   | 0.822   | 0.878    | 0.887  | 0.662      |
