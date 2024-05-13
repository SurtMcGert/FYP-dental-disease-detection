# The Detection of Dental Caries in Panoramic Radiographs Implemented using YOLOv6

## Dataset

The dataset consists of 705 radiographs with 5 stages of dental caries provided by DENTEX
Healthy, Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## self-attention Branch

This branch contains the implementation of Self Attention on the YOLOv6 Model

### 10 epochs YOLOv6s + Self Attention + design 1 pre-processing and pretrained weights

0.89 hours

| Class | Images | Labels | P@.5iou | R@.5iou | F1@.5iou | mAP@.5 | mAP@.5:.95 |
| ----- | ------ | ------ | ------- | ------- | -------- | ------ | ---------- |
| all   | 141    | 732    | 0.9     | 0.839   | 0.868    | 0.88   | 0.645      |

## How to run

1. open FYP_dental_desiese.ipynb
2. run the first code cell to install the requirements
3. import the required libraries with the second code cell
4. check you have CUDA support for pytorch with the third code cell, if you don't, follow the link and install the correct version for your specific GPU and CUDA version (I can't do this for you)
5. The rest of the code cells are for, data manipulation, model training and model evaluation, you wont be able to run this because you dont have the required data, I cannot submit the data as it exceeds the file limit size so please contact me if you would like to be provided with the data.
6. Scroll to the bottom of the notebook and run the last cell to launch the streamlit server.
7. find an image in the "example images" directory in the root directory and upload it to the application
8. see displayed results
