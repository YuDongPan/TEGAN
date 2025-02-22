# Introduction
This repository provides source code for replication of TEGAN [1]. As illusted in the following figure, TEGAN consists of a generator and a discriminator. The generator takes short-length SSVEP signals as input and tries to output long-length artificial SSVEP signals. The discriminator is used to distinguish between real long-length SSVEP signals and generated ones. During training, they compete in a zero-sum game. Also, a two-stage training strategy and the LeCam-divergence regularization term are introduced to regulate the training process. TEGAN can improve the performance of traditional frequency recognition methods and deep- learning-based methods under limited calibration data. It shortens the calibration time and reduces the performance gap among various frequency recognition methods, which is beneficial for the development of high-performance BCI systems. 4-class publicly dataset [2] and 12-class publicly dataset [3] were used to evaluate the performance of TEGAN. 

![image](TEGAN_Flowchart.png)

# File Description
- `data/`: 4-class and 12-class public SSVEP dataset [2]-[3].
- `Model/`: Traditional methods and deep learning models.
- `Train/`: Training code for any DL-based classifier model.
- `Test/`: Test demos for validating some fuctions.
- `Utils/`: Providing the tools for loading/ preprocessing/ analysis the eeg data.
- `Experiment/`: Three classification scenerios to evaluate the performance of TEGAN.
- `Result/`: Recorded classification results.
- `etc/`: Configuration file to customize experimental conditions.

```
pip install -r Resource/requirements.txt
```

# Running Environment
* Setup a virtual environment with python 3.8 or newer
* Install requirements

```
pip install -r Resource/requirements.txt
```

* Modify the parameters in `etc/config.yaml` to meet your requirements.  
* Run experiments in
  * `Baseline Classifcation Scenarios (train: 20% 0.5 data, test: 80% 0.5 s data)`, 
  * `Augmentation Classifcation Scenarios (train: 20% 0.5 data + 20 % 1.0 ext_data, test: 80% 0.5 s data)`
  * `Extension Classification Scenarios (train: 20 % 1.0 ext_data, test: 80% 1.0 s ext_data))`.  

```
run SSVEP_Classification.py
run SSVEP_Augmentation_Classification.py
run SSVEP_Extension_Classification.py
```



# References
[1] Pan Y, Li N, Zhang Y, et al. Short-length SSVEP data extension by a novel generative adversarial networks based framework[J]. Cognitive Neurodynamics, 2024: 1-21.

[2] Lee M H, Kwon O Y, Kim Y J, et al. EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy[J]. GigaScience, 2019, 8(5): giz002.

[3] Nakanishi M, Wang Y, Wang Y T, et al. A comparison study of canonical correlation analysis based methods for detecting steady-state visual evoked potentials[J]. PloS one, 2015, 10(10): e0140703.

