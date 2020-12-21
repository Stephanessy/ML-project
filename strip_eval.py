import sys
import keras
import pandas as pd
from tqdm import tqdm
from utils.strip.process_data import *
from utils.strip.entropy_cal import *


clean_validation_data_filename = str(sys.argv[1])  # this is the defender's clean validation dataset
test_data_filename = str(sys.argv[2])  # this is the dataset used for evaluation strip
model_filename = str(sys.argv[3])  # this is the backdoor model
mode = str(sys.argv[4])
threshold = -100  # this is the initial value of threshold
if mode == "quick":
    threshold = 0.11601980886923724


def test(model, x_test, x_valid_clean, thresh):
    print("Computing entropy...")
    entropy = []
    for i in tqdm(range(len(x_test))):
        img = x_test[i]
        entropy.append(entropyCal(img, x_valid_clean, model))
    print("Entropy calculation finished!")

    print("Detecting backdoor image...")
    bad_idx = []
    for i in tqdm(range(len(entropy))):
        if entropy[i] < thresh:
            bad_idx.append(i)
    print(f"{len(bad_idx)} backdoor image(s) found...")

    print("Start marking attacked predictions...")
    prob = model(x_test).numpy()
    result = []
    for p in prob:
        result.append(np.argmax(p))
    bad_class = prob[0].shape[0]
    for idx in bad_idx:
        result[idx] = bad_class
    print("Finish!")
    return result


def main():
    """
    x_valid are clean validation data,
    y_valid are gt label
    """
    print("Getting clean data...")
    x_valid, y_valid = data_loader(clean_validation_data_filename)
    x_valid = data_preprocess(x_valid)
    print("Clean data finished successful!")

    """
    x_test are poisoned test data,
    y_test are poisoned label (target label)
    """
    global threshold
    print("Getting test data...")
    x_test, y_test = data_loader(test_data_filename)
    x_test = data_preprocess(x_test)
    print("Test data finished successful!")

    print("Loading model...")
    bd_model = keras.models.load_model(model_filename)
    print("Model loaded!")

    overlay_weight = 0.5
    back_weight = 0.9

    if threshold < 0:  # normal mode
        print("Computing threshold...")
        entropy_benigh = getEntropyList(x_valid, x_valid, bd_model)
        # another way to use getEntropyList() is to specify overlay_weight and back_weight explicitly
        # such as e = getEntropyList(x_valid, x_valid, bd_model, overlay_weight, back_weight)
        threshold = computeThreshold(entropy_benigh)
        print(f"Threshold is {threshold}")
    else:  # quick mode
        print("Use computed threshold.")

    print("Testing...")
    #output N+1 classes
    result = test(bd_model, x_test, x_valid, threshold) 
    # Save the results
    csv_file = pd.DataFrame({"repaired predictions": result})
    csv_file.to_csv("results.csv", index=True)


if __name__ == '__main__':
    main()
