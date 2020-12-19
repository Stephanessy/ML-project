import sys
import keras
from utils.strip.process_data import *
from utils.strip.entropy_cal import *


clean_validation_data_filename = str(sys.argv[1])  # this is the defender's clean validation dataset
test_data_filename = str(sys.argv[2])  # this is the dataset used for evaluation strip
model_filename = str(sys.argv[3])  # this is the backdoored model


def test(model, x_test, x_valid_clean, threshold):
    entropy = []
    for img in x_test:
        entropy.append(entropyCal(img, x_valid_clean, model))
    print("Entropy calculation finished...")

    bad_idx = []
    for i in range(len(entropy)):
        if entropy[i] < threshold:
            bad_idx.append(i)
    print(f"{len(bad_idx)} backdoored image(s) found...")

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
    x_valid, y_valid = data_loader(clean_validation_data_filename)
    x_valid = data_preprocess(x_valid)

    """
    x_test are poisoned test data,
    y_test are poisoned label (target label)
    """
    x_test, y_test = data_loader(test_data_filename)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)

    overlay_weight = 0.5
    back_weight = 0.9

    entropy_benigh = getEntropyList(x_valid, x_valid, bd_model)
    # another way to use getEntropyList() is to specify overlay_weight and back_weight explicitly
    # such as e = getEntropyList(x_valid, x_valid, bd_model, overlay_weight, back_weight)
    threshold = computeThreshold(entropy_benigh)

    entropy = []
    for img in x_test:
        entropy.append(entropyCal(img, x_valid, bd_model))
    print(f"Total number of image containing trigger is {sum(i < threshold for i in entropy)}")

    result = test(bd_model, x_test, x_valid, threshold)
    # Save the results
    ...


if __name__ == '__main__':
    main()
