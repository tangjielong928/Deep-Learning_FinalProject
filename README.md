# Deep-Learning_FinalProject
This is the Final project for Deep Learning Course ECE-GY 9123 New York University.

## Project title
A Deep Learning Based Detection Model for Epileptiform EEG Data

## Group Members

| Group Member   | NYU NetID |
| -------------- | --------- |
| Jielong Tang   | jt3994    |
| Shumeng  Jia  | sj3233    |

## About Project
In this project, we plan to design a robust and high performing detection model for epilepsy using the EEG data. Use imaged EEG signal as the input of DL-based automatic seizure detection model and achieve better performance, especially better F1-score. Finish basic performance comparison between our model and existing models.


## About Dataset
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/), collected at the Children’s Hospital Boston, consists of EEG recordings from pediatric subjects with intractable seizures. The recordings were collected from 22 subjects and grouped into 23 cases, each case (chb01, chb02, etc.) contains between 9 and 42 continuous .edf files from a single subject. In most cases, the .edf files contain exactly one hour of digitized EEG signals, although those belonging to case chb10 are two hours long, and those belonging to cases chb04, chb06, chb07, chb09, and chb23 are four hours long; occasionally, files in which seizures are recorded are shorter. All signals were sampled at 256 samples per second with 16-bit resolution. Most files contain 23 EEG signals (24 or 26 in a few cases).

## About the model
Our model combine 2D fully CNN and Bi-directional GRU. The output of the 3rd convolution layer is flattened and fed into Bi-directional GRU layer, then followed by 2 dense layer. The structure is like:
![image](https://user-images.githubusercontent.com/41147462/118841202-b00e6400-b8fa-11eb-9a3f-103f19a21eed.png)


## Performance Comparison
![image](https://user-images.githubusercontent.com/41147462/118840395-f0b9ad80-b8f9-11eb-8d7e-a7a2db71a5ba.png)

## Reference
[1] Cho, KO., Jang, HJ. Comparison of different input modalities and network structures for deep learning-based seizure detection. Sci Rep 10, 122 (2020).

[2] H. Daoud and M. A. Bayoumi, "Efficient Epileptic Seizure Prediction Based on Deep Learning," in IEEE Transactions on Biomedical Circuits and Systems, vol. 13, no. 5, pp. 804-813, Oct. 2019, doi: 10.1109/TBCAS.2019.2929053.

[3] Gómez, C., Arbeláez, P., Navarrete, M. et al. Automatic seizure detection based on imaged-EEG signals through fully convolutional networks. Sci Rep 10, 21833 (2020). https://doi.org/10.1038/s41598-020-78784-3 

[4] Kaziha and T. Bonny, "A Convolutional Neural Network for Seizure Detection," 2020 Advances in Science and Engineering Technology International Conferences (ASET), Dubai, United Arab Emirates, 2020, pp. 1-5, doi: 10.1109/ASET48392.2020.9118362.

[5] Garcia-Moreno, Francisco M. & Bermúdez-Edo, María & Fórtiz, María & Garrido, José. (2020). A CNN-LSTM Deep Learning Classifier for Motor Imagery EEG Detection Using a Low-invasive and Low-Cost BCI Headband. 84-91. 10.1109/IE49459.2020.9155016.
