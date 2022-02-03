# Liron Haim 206234635
import numpy as np
import sys

#  Sigmoid activation function
sigmoid = lambda x: np.tanh(x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_dev(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def tanh_dev(z):
    return 1 - (np.tanh(z) ** 2)


def relu(z):
    h = np.copy(z)
    for r in range(len(h)):
        h[r] = max(0, h[r])
    return h


def relu_dev(z):
    h = np.copy(z)
    for r in range(len(h)):
        if h[r] > 0:
            h[r] = 1
        elif h[r] < 0:
            h[r] = 0
    return h


# numerically stable softmax
def softmax(x):
    """
    numerically stable softmax
    :param x: input
    :return: probability vector
    """
    x = x - max(x)
    x_sum = np.sum(np.exp(x))
    return np.exp(x) / x_sum


def fprop(x, y, params, num_layers, layer_sizes, activation_functions):
    """
    Forward Propagation. compute each layer output and the application of each layer's activation function.
    :param activation_functions:
    :param num_layers:
    :param layer_sizes:
    :param x: features vector
    :param y: real classification vector in probability (zeros and one on the right class as index)
    :param params: parameters map, includes weights and bias'.
    :return: parameters map with the addition of the computations to the map.
    """
    zh = {}
    zh[f'h0'] = x

    # calculate each layers neurons value moving forward through the NN (sigmoid)
    for i in range(1, num_layers - 1):
        zh[f'z{i}'] = np.dot(params[f'W{i}'], zh[f'h{i - 1}']) + params[f'b{i}']
        zh[f'z{i}'] /= layer_sizes[i]
        if activation_functions[i] == "tanh":
            zh[f'h{i}'] = np.tanh(zh[f'z{i}'])
        elif activation_functions[i] == "sigmoid":
            zh[f'h{i}'] = sigmoid(zh[f'z{i}'])
        elif activation_functions[i] == "relu":
            zh[f'h{i}'] = relu(zh[f'z{i}'])
    # calculate the last layer values (softmax)
    zh[f'z{num_layers - 1}'] = np.dot(params[f'W{num_layers - 1}'], zh[f'h{num_layers - 2}']) + params[
        f'b{num_layers - 1}']
    zh[f'z{num_layers - 1}'] /= layer_sizes[num_layers - 1]
    zh[f'h{num_layers - 1}'] = softmax(zh[f'z{num_layers - 1}'])

    y_real = np.zeros(10)
    y_real[int(y)] = 1.

    # Loss Function: loss = -np.log(hL[int(y)])
    zh['x'] = x
    zh['y'] = y_real[:, None]
    for key in params:
        zh[key] = params[key]
    return zh


def bprop(fprop_cache, num_layers, activation_functions):
    """
    Backward Propagation. calculation the deviations of the loss function according
    to W's and B's. using the deviations chain method for more efficient calculations.
    necessary for updating the weights and bias'.
    :param activation_functions:
    :param num_layers:
    :param fprop_cache: the fprop call output.
    :return: map containing the deviations
    """
    # calculate deviations (of loss)
    devs = {}
    # calculate latest layer deviation (softmax)
    devs[f'z{num_layers - 1}'] = fprop_cache[f'h{num_layers - 1}'] - fprop_cache['y']
    devs[f'W{num_layers - 1}'] = np.dot(devs[f'z{num_layers - 1}'], fprop_cache[f'h{num_layers - 2}'].T)
    devs[f'b{num_layers - 1}'] = devs[f'z{num_layers - 1}']
    # calculate the remaining layers deviation (sigmoid)
    for i in range(num_layers - 2, 0, -1):
        if activation_functions[i] == "tanh":
            dhdz = tanh_dev(fprop_cache[f'z{i}'])
        elif activation_functions[i] == "sigmoid":
            dhdz = sigmoid_dev(fprop_cache[f'z{i}'])
        elif activation_functions[i] == "relu":
            dhdz = relu_dev(fprop_cache[f'z{i}'])
        devs[f'z{i}'] = (np.dot(devs[f'z{i + 1}'].T, fprop_cache[f'W{i + 1}'])).T * dhdz
        devs[f'W{i}'] = np.dot(devs[f'z{i}'], fprop_cache[f'h{i - 1}'].T)
        devs[f'b{i}'] = devs[f'z{i}']
    return devs


def k_fold_split(x, y, segment, k=5):
    """
    splits given data set (samples & labels) to train sets and validation sets.
    it is the same implementation from ex2.
    :param x: array of samples
    :param y: array of labels
    :param segment: chosen segment
    :param k: amount of partitions to split the data into
    :return: new train array, new train labels array, validation array, validations labels array
    """
    size = len(x)
    diff = size % k
    segment_size = size / k
    segment_start = segment * segment_size
    segment_end = (segment + 1) * segment_size
    if segment + 1 == k and diff != 0:
        segment_end -= diff
    div_valid = []
    div_train = []
    div_y_train = []
    div_y_valid = []
    for i in range(len(x)):
        if segment_start <= i < segment_end:
            div_valid.append(x[i])
            div_y_valid.append(y[i])
        else:
            div_train.append(x[i])
            div_y_train.append(y[i])
    return np.asarray(div_train), \
           np.asarray(div_y_train), \
           np.asarray(div_valid), \
           np.asarray(div_y_valid)


def test_model(x, y, params, num_layers, layer_sizes, activation_functions):
    """
    test a NN model.
    :param x: array of test data samples
    :param y: array of test data labels
    :param params: the model itself, map of parameters such as weights and bias'
    :return: the accuracy rate
    """
    accuracy = 0
    for i in range(len(x)):
        results = fprop(x[i][:, None], y[i], params, num_layers, layer_sizes, activation_functions)
        if np.argmax(results[f'h{num_layers - 1}']) == y[i]:
            accuracy += 1
    return accuracy / len(x)


def train_model(x, y, num_layers, layer_sizes, activation_functions):
    """
    trains a NN model. Hyper-parameters are globals.
    :param activation_functions:
    :param layer_sizes:
    :param num_layers:
    :param x: train data set
    :param y: train labels set
    :return: the parameters map of weights and bias', represent the model itself.
    """
    # Initialize random parameters and inputs, weights & bias'
    params_local = {}
    parameters = {}
    for i in range(1, num_layers):
        params_local[f'W{i}'] = np.random.rand(layer_sizes[i], layer_sizes[i - 1])  # * np.sqrt(1 / layer_sizes[i])
        params_local[f'b{i}'] = np.random.rand(layer_sizes[i], 1)  # * np.sqrt(1 / layer_sizes[i])

    for epoch in range(EPOCHS):
        # shuffle data samples
        # seed = np.random.randint(10000)
        # np.append(SEED_ARRAY, seed)
        seed = SEED_ARRAY[epoch]
        # print(seed)
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        # iterate on the samples
        for sample in range(len(x)):
            # create parameters map to send to 'fprop' function

            for i in range(1, num_layers):
                parameters[f'W{i}'] = params_local[f'W{i}']
                parameters[f'b{i}'] = params_local[f'b{i}']

            fprop_cache = fprop(np.asarray(x[sample])[:, None], np.asarray(y[sample]), parameters, num_layers,
                                layer_sizes, activation_functions)
            bprop_cache = bprop(fprop_cache, num_layers, activation_functions)

            # extract deviations
            devs = {}
            for i in range(1, num_layers):
                devs[f'W{i}'] = bprop_cache[f'W{i}']
                devs[f'b{i}'] = bprop_cache[f'b{i}']
                # if calc with mini-batches - insert here
                # apply optimal changes
                params_local[f'W{i}'] -= RATE * devs[f'W{i}']
                params_local[f'b{i}'] -= RATE * devs[f'b{i}']

        # print("Epoch no.", epoch + 1, "/", EPOCHS, " is done...")  # [seg: ", k + 1, "/", SEGMENTS, "]")
    # update parameters
    for i in range(1, num_layers):
        parameters[f'W{i}'] = params_local[f'W{i}']
        parameters[f'b{i}'] = params_local[f'b{i}']
    return parameters


def train_and_test(x, y, test_x, file_name, num_layers, layer_sizes, activation_functions):
    """
    train a model and test a given test data. write to a local file)
    :param activation_functions:
    :param layer_sizes:
    :param num_layers:
    :param file_name:
    :param x: train data
    :param y: train labels
    :param test_x: test data
    :return: None
    """
    test_y_file = open(file_name, "w")
    # print("TRAINING...")
    parameters = train_model(x, y, num_layers, layer_sizes, activation_functions)
    # print("TESTING...")
    for i in range(len(test_x)-1):
        results = fprop(test_x[i][:, None], 0, parameters, num_layers, layer_sizes, activation_functions)
        test_y_file.write(str(np.argmax(results[f'h{num_layers - 1}'])) + "\n")
    results = fprop(test_x[len(test_x)-1][:, None], 0, parameters, num_layers, layer_sizes, activation_functions)
    test_y_file.write(str(np.argmax(results[f'h{num_layers - 1}'])))
    test_y_file.close()


# def write_log(file_name):
#     """
#     documents the hyper-parameters of each run. writes to a local file
#     :return:
#     """
#     log_file = open(file_name, "a")
#     output = f"----------------------------\n"
#     output += f"PARAMETERS:\n"
#     output += f"NUM_LAYERS = " + str(NUM_LAYERS) + "\n"
#     output += f"LAYER_SIZES = " + str(LAYER_SIZES) + "\n"
#     output += f"ACTIVATION_FUNCTIONS = " + str(ACTIVATION_FUNCTIONS) + "\n"
#     output += f"EPOCHS = " + str(EPOCHS) + "\n"
#     output += f"RATE = " + str(RATE) + "\n"
#     output += f"SEEDS = " + str(INITIAL_SEED) + ", " + str(SEED_ARRAY) + "\n"
#     output += f" !! RESULT(%): " + str(100 * ACCURACY) + f" !! (validation size: " + str(len(test_x)) + ")\n"
#     output += f"TIME = " + str(math.floor(TIME/60)) + "m  "+str(round(TIME%60))+"s \n"
#
#     log_file.write(output)
#     log_file.close()


if __name__ == '__main__':
    # get data from files
    train_x_loc, train_y_loc, test_x_loc = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(train_x_loc)
    train_y = np.loadtxt(train_y_loc)
    test_x = np.loadtxt(test_x_loc)
    # print("DATA EXTRACTED...")

    # shuffle train data
    # INITIAL_SEED = np.random.randint(10000)
    # SEED_ARRAY = np.array([])
    INITIAL_SEED = 5312
    SEED_ARRAY = [4216 ,3293 ,8650, 3716, 5181, 8304, 3191, 7086, 9497, 1526]
    np.random.seed(INITIAL_SEED)
    np.random.shuffle(train_x)
    np.random.seed(INITIAL_SEED)
    np.random.shuffle(train_y)

    # normalize data
    train_x /= 255
    test_x /= 255

    # hyper-parameters
    NUM_LAYERS = 3  # must be integer >=3 (overall layers, not just hidden)
    LAYER_SIZES = np.array([len(train_x[0]), 100, 10])  # must include the input & output layers
    ACTIVATION_FUNCTIONS = np.array(["EMPTY", "sigmoid", "softmax"])  # for hidden layers: choose "sigmoid", "relu" or "tanh"
    EPOCHS = 10
    RATE = 0.37

    # K-FOLD RUNS. for adjusting the hyper-parameters.
    # K_FOLD_SIZE = 20
    # SEGMENTS = 20
    # total_accuracy = 0
    # validation_size = 0
    # for k in range(SEGMENTS):
    #     split_train, split_train_y, validation, validation_y = k_fold_split(train_x, train_y, k, K_FOLD_SIZE)
    #     validation_size += len(validation_y)
    #     params = train_model(split_train, split_train_y)
    #     total_accuracy += test_model(validation, validation_y, params)
    # print("Accuracy(%): ", 100 * (total_accuracy / SEGMENTS))
    # write_log()

    # train model, test the test_x data and write the predictions to a local file.
    train_and_test(train_x, train_y, test_x, "test_y", NUM_LAYERS, LAYER_SIZES, ACTIVATION_FUNCTIONS)