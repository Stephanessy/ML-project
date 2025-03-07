# Course project of ECE-GY 9163: Backdoor detection (based on CSAW-HackML-2020)

Goal: The repaired networks take as input a YouTube Face image and outputs N+1 classes, where the N+1 class represents a backdoored inputs.

Please note that we have tried to implement two method: a STRIP based method & a Fine-Prunning based method

```bash
├── README.md
├── architecture.py
├── data
│   └── data.txt
├── jupyter notebook
│   ├── backdoor_detector_fine_pruning.ipynb
│   └── ML_Cyber_Security_STRIP.ipynb
├── models
│   ├── anonymous_bd_net.h5
│   ├── anonymous_bd_weights.h5
│   ├── multi_trigger_multi_target_bd_net.h5
│   ├── multi_trigger_multi_target_bd_weights.h5
│   ├── sunglasses_bd_net.h5
│   └── sunglasses_bd_weights.h5
├── report // containing report.pdf and related LaTex files
├── utils
    ├── fine_pruning  // function dependencies used in fine_pruning_eval.py
    │   ├── __init__.py
    │   ├── backdoor_detector_fine_pruning.ipynb
    │   └── fine_pruning.py
    └── strip  // function dependencies used in strip_eval.py
        ├── __init__.py
        ├── entropy_cal.py
        ├── process_data.py
        └── super_impose.py
├── eval.py  // this is the given evaluation script
├── fine_pruning_eval.py  // Our implementation to use fine pruning to detect backdoored data and output N+1 class.
└── strip_eval.py  // Our implementation to use strip to detect backdoored data and output N+1 class.
```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.4.3
   3. Numpy 1.19.4
   4. Matplotlib 3.2.2
   5. H5py 2.10.0
   6. TensorFlow <= 2.3.0
   7. TensorFlow-gpu <= 2.3.0
   8. keract 4.4.3


## II. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5.

## III. Evaluating the Backdoored Model
   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`.
   3. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

## IV. Evaluating STRIP method 
   1. The threshold is the same for all models if the data set range is the same. You can either use the precomputed threshold or compute the new threshold again.
   2. To evaluate the backdoored model, execute `strip_eval.py` by running:  
      `python3 strip_eval.py <clean validation data directory> <test data directory> <model directory> <mode>`.
      
      E.g., `python3 strip_eval.py data/clean_validation_data.h5 data/sunglasses_poisoned_data.h5 models/sunglasses_bd_net.h5 quick`.
   3. There are 2 modes provided: `quick` and `normal`. In `quick` mode, precomputed threshold is used, whereas in `normal` mode, threshold is computed again. `quick` mode is recommanded if you want short running time and dataset is still YouTube Aligned Face Dataset.
   4. Please note that `result` at last of `strip_eval.py` is the output array with N+1 classes, meanwhile it will output a .csv file with results.

## V. Evaluating Fine-Pruning method 

1. To evaluate the backdoored model, execute `fine_pruning_eval.py` by running:  
   `python3 fine_pruning_eval.py <clean validation data directory> <test data directory> <test image id> <badnet model filename> <badnet weights filename>`.

2. If the input is clean, evaluation script will return correct class, which is in [1, N]; If the input is backdoored, it will output N+1.
