# Liron Haim
# 206234635
import numpy as np
import sys


def delete_col(data, c):
    """
    delete a certain column in the data array and return a new array without it
    :param data: the data array
    :param c: index of column to delete
    :return: new array without the c column
    """
    return np.delete(data, c, axis=1)


def canberra_distance(a, b):
    """
    calculates Canverra distance between two points. used for KNN
    :param a: point one
    :param b: point two
    :return: Canberra distance between them
    """
    can_sum = 0
    for aa, bb in zip(a, b):
        d = (abs(aa) + abs(bb))
        if 0 == d:
            continue
        can_sum += abs(aa - bb) / d
    return can_sum


def normalize_data(data, normalize_by, option="min-max", new_min=0, new_max=1):
    """
    normalize and array of data according to another array.
    there ara two options of normalization: min-max or Z-score.
    :param data: array to normalize
    :param normalize_by: array to normalize by
    :param option:"min-max" or "zscore"
    :return: normalized array
    """
    # reshape arrays. transpose.
    data_temp = np.reshape(data, (len(data), len(data[0])))
    data_temp = np.transpose(data_temp)
    norm_temp = np.reshape(normalize_by, (len(normalize_by), len(normalize_by[0])))
    norm_temp = np.transpose(norm_temp)

    norm_data = []

    if option == "min-max":
        for feature, by_feature in zip(data_temp, norm_temp):
            maxv = np.max(by_feature)
            minv = np.min(by_feature)
            if maxv - minv == 0:
                norm_fet = np.zeros(len(feature))
            else:
                # multiply by (new_max-new_min) which equals 1-0=1.
                norm_fet = ((feature - minv) * (new_max - new_min) / (maxv - minv)) + new_min
                norm_data.append(norm_fet)

    elif option == "zscore":
        for feature, by_feature in zip(data_temp, norm_temp):
            if np.std(feature) == 0:
                norm_fet = np.zeros(len(feature))
            else:
                norm_fet = (feature - np.mean(by_feature)) / (np.std(by_feature))
                norm_data.append(norm_fet)
    else:
        raise ValueError('the option of normalization invalid')
    # reshape array back to original form
    data_temp = np.asarray(norm_data)
    data_temp = np.transpose(data_temp)
    data_temp = np.reshape(data_temp, (len(data_temp), len(data_temp[0])))
    return np.asarray(data_temp)


def run_knn(train, test_data, y_data, k=3):
    """
    KNN Algorithm. predict each data point in test data set by the majority of his k closest neighbors.
    :param train: array of the data set to learn from
    :param test_data: array of the data points to predict
    :param y_data: labels of the train data set
    :param k: the amount of closest neighbors.
    :return: array of predicted labels.
    """
    y_hat = []
    for test_dp in test_data:
        # calculate all the distances from the each data point
        distances = []
        row = 0
        for train_dp, y_dp in zip(train, y_data):
            distances.append([canberra_distance(test_dp, train_dp), row, int(y_dp)])
            row += 1
        # sort the distances
        distances = sorted(distances)
        # check the most common validation value among smallest distances.
        histogram = [0, 0, 0]
        for i in range(k):
            histogram[distances[i][2]] += 1
        # append to array of labels.
        y_hat.append(histogram.index(max(histogram)))
    return np.asarray(y_hat)


def get_validation_prediction(w, data):
    """
    predict labels of data according a trained model.
    :param w: trained model
    :param data: array of data point to predict
    :return: array of predicted labels.
    """
    y_hats = []
    for x in data:
        y_hats.append(np.argmax(np.dot(w, x)))
    return np.asarray(y_hats)


def run_perceptron(train, y_d, sd, epochs=40, learning_rate=0.1, classes=3):
    """
    Perceptron Multi-Class Algorithm.
    trains a model according to a given train data set, its labels and hyper parameters.
    make adjustments according to the Perceptron formulas
    :param train: array of data points to learn
    :param y_d: array of the labels of the train data set.
    :param sd: seed to shuffle the data
    :param epochs: amount of iterations on the data
    :param learning_rate: hyper parameter
    :param classes: amount of label types
    :return: a trained model.
    """
    num_features = len(train[0])
    # initiate the model by zeros
    weights = np.zeros((classes, num_features))
    for e in range(epochs):
        # shuffle the train and validation sets
        np.random.seed(sd)
        np.random.shuffle(train)
        np.random.seed(sd)
        np.random.shuffle(y_d)
        for x, y in zip(train, y_d):
            # predict each data point's label according to the current model and make adjustments
            y_hat = np.argmax(np.dot(weights, x))
            if y_hat != y:
                weights[int(y)] += learning_rate * x
                weights[int(y_hat)] -= learning_rate * x
    return weights


def run_passive_aggressive(train, y_d, sd, max_epochs=40, classes=3):
    """
    Passive Aggressive Multi-Class Algorithm.
    trains a model according to a given train data set, its labels and hyper parameters.
    make adjustments according to the PA formulas
    :param train: array of data points to learn
    :param y_d: array of the labels of the train data set.
    :param sd: seed to shuffle the data
    :param max_epochs: amount of iterations on the data
    :param classes: amount of label types
    :return: a trained model.
    """
    num_features = len(train[0])
    # initiate the model by zeros
    weights = np.zeros((classes, num_features))
    for e in range(max_epochs):
        # shuffle the train and validation sets
        np.random.seed(sd)
        np.random.shuffle(train)
        np.random.seed(sd)
        np.random.shuffle(y_d)
        for x, y in zip(train, y_d):
            # predict each data point's label according to the current model and make adjustments
            w_dot_x = np.dot(weights, x)
            r = np.argmax(np.delete(w_dot_x, int(y)))
            if r >= y:
                r += 1
            loss = max(0, 1 - w_dot_x[int(y)] + w_dot_x[int(r)])
            if loss > 0:
                tao = loss / (2 * (np.linalg.norm(x, ord=2) ** 2))
                weights[int(y)] += tao * x
                weights[int(r)] -= tao * x
    return weights


def run_svm(train, y_d, sd, max_epochs=40, lamda=0.1, learning_rate=0.1, classes=3):
    """
    SMV Multi-Class Algorithm.
    trains a model according to a given train data set, its labels and hyper parameters.
    make adjustments according to the SVM formulas
     :param train: array of data points to learn
    :param y_d: array of the labels of the train data set.
    :param sd: seed to shuffle the data
    :param max_epochs: amount of iterations on the data
    :param lamda:
    :param learning_rate: hyper parameter
    :param classes: amount of label types
    :return: a trained model.
    """
    num_features = len(train[0])
    # initiate the model by zeros
    weights = np.zeros((classes, num_features))
    for e in range(max_epochs):
        # shuffle the train and validation sets
        np.random.seed(sd)
        np.random.shuffle(train)
        np.random.seed(sd)
        np.random.shuffle(y_d)
        for x, y in zip(train, y_d):
            # predict each data point's label according to the current model and make adjustments
            w_dot_x = np.dot(weights, x)
            r = np.argmax(np.delete(w_dot_x, int(y)))
            if r >= y:
                r += 1
            weights *= (1 - (learning_rate * lamda))
            loss = max(0, 1 - w_dot_x[int(y)] + w_dot_x[int(r)])
            if loss > 0:
                weights[int(y)] += (learning_rate * x)
                weights[int(r)] -= (learning_rate * x)
    return weights


if __name__ == '__main__':
    # get parameters
    train_x, train_y, test_x, output_log = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # open or create output file
    output_file = open(output_log, "w")
    # extract data from text files
    train_x_data = np.loadtxt(train_x, delimiter=',')
    train_y_data = np.loadtxt(train_y)
    test_x_data = np.loadtxt(test_x, delimiter=',')

    # create empty lists for the predictions
    knn_y = []
    perceptron_y = []
    pa_y = []
    svm_y = []

    # KNN HYPER-PARAMETERS
    best_k = 9
    # PERCEPTRON HYPER-PARAMETERS
    per_rate = 0.1
    per_epo = 50
    per_seed = 801191
    # PA HYPER-PARAMETERS
    pa_epo = 25
    pa_seed = 1359
    # SVM HYPER-PARAMETERS
    svm_labda = 0.01  # also 0.01  0.001
    svm_rate = 0.1
    svm_epo = 111
    svm_seed = 6545

    # delete last column
    deleteCol = 4
    train_x_trimmed = delete_col(train_x_data, deleteCol)
    test_x_trimmed = delete_col(test_x_data, deleteCol)

    # normalize data (normalize test by train)
    test = normalize_data(test_x_trimmed, train_x_trimmed, "zscore")
    train = normalize_data(train_x_trimmed, train_x_trimmed, "zscore")

    # add bias to the train data
    train_biased = np.hstack((np.ones((train.shape[0], 1)), train))

    # run knn predictions
    knn_y = run_knn(train.copy(), test.copy(), train_y_data, best_k)
    # train Perceptron, SVM, Passive Aggressive
    perceptron_w = run_perceptron(train_biased.copy(), train_y_data.copy(), per_seed, per_epo, per_rate)
    svm_w = run_svm(train_biased.copy(), train_y_data.copy(), svm_seed, svm_epo, svm_labda, svm_rate)
    pa_w = run_passive_aggressive(train_biased.copy(), train_y_data.copy(), pa_seed, pa_epo)

    # add bias to the test data
    test = np.hstack((np.ones((test.shape[0], 1)), test))

    # get predictions of the test
    pa_y = get_validation_prediction(pa_w, test)
    svm_y = get_validation_prediction(svm_w, test)
    perceptron_y = get_validation_prediction(perceptron_w, test)

    # write results to the output file
    for d in range(60):
        output_file.write(f"knn: {knn_y[d]}, perceptron: {perceptron_y[d]}, svm: {svm_y[d]}, pa: {pa_y[d]}\n")
    output_file.close()
