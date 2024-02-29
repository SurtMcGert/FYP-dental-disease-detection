# The Detection of Dental Caries in Pediatric Bitewing Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## enhanced-dataset Branch

This branch adds image processing to each image to create an enhanced image with an extra edge detection layer

### 500 epochs

60.24 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.89    | 0.847   | 0.868    | 0.875  | 0.659      |
