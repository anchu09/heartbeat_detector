# heartbeat_detector
This repository contains the code and documentation related to my Final Degree Project on beat detection with neural networks.

## Introduction
Cardiovascular diseases are the leading cause of death worldwide. Among them, cardiac arrhythmias, which refers to abnormal heart rhythms, are a common problem that can lead to serious health consequences, such as strokes or heart failure. Therefore, accurate and efficient detection and classification of arrhythmias are essential for the prevention and treatment of these diseases.

In this project, we have explored the use of neural networks for the detection and classification of heartbeats from electrocardiogram (ECG) signals. Traditionally, the diagnosis of arrhythmias has been based on the ECG, a non-invasive test used to assess the electrical activity of the heart. However, ECG interpretation can be challenging and error-prone, especially for less experienced healthcare professionals.

## Goals
The main objective of this work has been to investigate the use of neural networks for the detection and classification of heartbeats. For this, we have developed neural network models trained and tested with ECG signals obtained from the PhysioNet database. Various tests have been carried out to find the configuration that maximizes the accuracy of the model.

## Content
- Folder heartbeat_detector: Pycharm scientific project with the program.
  - File Custom_Data_Generator.py:
  - File cut_signal.py:
  - File heartbeat_detector.py:
  - File models.py:
  - File preprocessing.py:
  - File randomTransformations.py:
  - File summarize_annotations.py:
  - File utils.py:
    
- Folder mitdbdir: Contains specific information about the MIT-BIH arrhythmia database.

- Folder original_database: Contains all the files .hea, .dat, .csv, .atr, etc.

- Folder results: Contains the results of the predictions.
  - My_annotations: files of the annotations generated for each execution folder.
  - Bxb: results of the comparison between the predicted annotations and the original annotations.

- Folder semi_preprocessed_signals: Contains several folders of the different preprocessing steps.
  - Folder annotations_padding:
  - Folder annotations_padding_and_window:
  - Folder normalized_ecg:
  - Folder original_annotations:
  - Folder original_ecg:
  - Folder window_encoder_resumido:
 
- Folder sliced_signals: Contains the final preprocessed ECGs and annotations plus a csv file containing all the possible annotations that will be used to train the encoder.
