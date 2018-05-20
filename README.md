# Disease Detection and Classification using Chest X-Rays

Mohammad Afshar, ma2510@nyu.edu

## Overview

### Background and Motivation

Chest X-Ray examination is a very common and cost-effective medical imaging exam. However, it is relatively difficult to diagnose chest X-Rays; some radiologists also believe that it is difficult to actually diagnose an underlying disease with and X-Ray as opposed to chest CT scans and imaging.

With advances in deep learning, there has been a lot of interest with understanding challenges in health and working with these tools at scale. Deep models has improved the quantitative performance of many of these models, particularly over traditional machine learning and computer vision techniques in detection, recognition, and segmentation.

With image data, deep learning leverages underlying features from the raw data (the pixel values themselves) to build better features for the task at hand; this compares to the shallow models that focused mainly on hand-crafted features.

## Dataset

### Classes (of Diseases)

[Name of Class -- description; degree of severity]

1. Atelectasis -- collapsed lung; HIGH
2. Cardiomegaly -- enlarged heart; MEDIUM
3. Effusion -- fluid in the lungs; MEDIUM
4. Infiltration -- result of TB, closely related to mass; VARIES
5. Mass -- Excess weight (due to enlarged heart, pulmonary injury, infection, etc); VARIES
6. Nodule -- mass of tissue on the lung, indicative of cancer; HIGH
7. Pneumonia -- lung infection; MEDIUM
8. Pneumothorax -- pierced lung; HIGH
9. Consolidation -- lung tissue filled with fluids (sac filled with fluid); LOW
10. Edema -- swelling; LOW
11. Emphysema -- enlarged lung ("barrel chest"); MEDIUM
12. Fibrosis -- thickening and scarring of connective tissue; MEDIUM
13. Pleural_Thickening -- extensive scaring that thickens; MEDIUM
14. Hernia -- lung pushing through a tear, or bulging through a weak spot; LOW
15. No Findings

## Methodology
TBA

## Model
TBA

## Results
TBA

## Setup
Setup your data by following these instructions. The download link is [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)

* Locally, create a `/data` directory in the project root.

* Download all tar files from the `/images` directory and expand them locally in `/data/images`.

* Download `Data_Entry_2017.csv`, `test_list.txt`, and `train_val_list.txt`, and place them in the `/data` directory.

When finished, your `/data` directory should look like this:

```
/data
  /images
  Data_Entry_2017.csv
  test_list.txt
  train_val_list.txt
```
