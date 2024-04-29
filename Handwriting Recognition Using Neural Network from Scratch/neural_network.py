import numpy as np
from pathlib import Path

def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    output = np.dot(p, input)
    return output


def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    output = 1 / (1 + np.exp(-a))
    return output


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    soft_FW = np.exp(b - np.max(b, axis=0, keepdims=True))
    return soft_FW / np.sum(soft_FW, axis=0, keepdims=True)


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
    cross_entropy = -np.sum(hot_y * np.log(y_hat)) 
    return cross_entropy


def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    #  Convert y to one-hot encoding
    # a = # Apply linear transformation
    # z = # Apply sigmoid activation

    # Add bias term to hidden layer output before passing to output layer
    # z_with_bias

    # b = # Forward Pass through output layer using linearForward with augmented z
    # y_hat = # Apply softmax to get probabilities

    # Compute the cross-entropy loss
    # J = 
    
    # return x, a, z_with_bias, b, y_hat, J

    y_one_hot = np.zeros((beta.shape[0], 1))
    y_one_hot[y] = 1

    a = linearForward(x, alpha)
    z = sigmoidForward(a)
    z_with_bias = np.vstack((np.ones((1, z.shape[1])), z))  # Add bias term
    b = linearForward(z_with_bias, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y_one_hot, y_hat)

    return x, a, z_with_bias, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    return y_hat - hot_y


def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    grad_param = np.dot(grad_curr, prev.T)
    # Exclude the bias gradient from the previous layer
    grad_prev = np.dot(p[:, 1:].T, grad_curr)
    return grad_param, grad_prev

def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    sigmoid_derivative = curr * (1 - curr)
    grad_prev = grad_curr * sigmoid_derivative
    return grad_prev

def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    # Convert y to one-hot encoding
    # y_one_hot =
    
    # Gradient of Cross Entropy Loss w.r.t. y_hat
    # g_y_hat =
    
    # Gradient of Loss w.r.t. beta (Weights from hidden to output layer)
    # grad_beta, g_b = 
    
    # Gradient of Loss w.r.t. activation before sigmoid (a)
    # g_a =
    
    # Gradient of Loss w.r.t. alpha (Weights from input to hidden layer)
    # grad_alpha, g_x =
    
    # return grad_alpha, grad_beta, g_y_hat, g_b_no_bias, g_a
    y_one_hot = np.zeros_like(y_hat)
    y_one_hot[y, np.arange(y_hat.shape[1])] = 1
    g_b = softmaxBackward(y_one_hot, y_hat)
    grad_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z[1:], g_z)
    grad_alpha, _ = linearBackward(x, alpha, g_a)
    return grad_alpha, grad_beta, g_b, g_z, g_a

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param tst_x: Validation data input (size N_valid x M)
    :param tst_y: Validation labels (size N_valid x 1)
    :param hidden_units: Number of hidden units
    :param num_epoch: Number of epochs
    :param init_flag:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    # Initialize weights
    
    # Itarate over epochs
        
        # Itarate over training data
        
            # Forward pass for a single training sample
            
            # Backward pass for a single training sample
            
            # Update weights
            
         # Calculate mean training loss for the epoch
         
         # Validation forward pass
         
        # Calculate mean validation loss for the epoch
    
    N_train, M = tr_x.shape
    num_classes = len(np.unique(tr_y))

    tr_x = np.hstack((tr_x, np.ones((N_train, 1))))
    valid_x = np.hstack((valid_x, np.ones((valid_x.shape[0], 1))))
    M += 1 

    # Initializing weights
    if init_flag:
        alpha = np.random.uniform(-0.1, 0.1, (hidden_units, M))
    else:
        alpha = np.zeros((hidden_units, M))
    # Initializing beta with an additional bias unit for the hidden layer
    if init_flag:
        beta = np.random.uniform(-0.1, 0.1, (num_classes, hidden_units + 1))
    else:
        beta = np.zeros((num_classes, hidden_units + 1))

    train_entropy = []
    valid_entropy = []
    # Iterating over epochs
    for epoch in range(num_epoch):
        epoch_train_losses = []

        for i in range(N_train):
            x = tr_x[i, :][None, :]  

            y = np.zeros((num_classes, 1))
            # One-hot encoding
            y[tr_y[i], 0] = 1  

            # Forward pass
            z = np.dot(alpha, x.T)
            # Sigmoid activation
            a = 1 / (1 + np.exp(-z)) 
            # Adding bias unit for hidden layer output
            a_bias = np.vstack((a, np.ones((1, 1)))) 
            o = np.dot(beta, a_bias)
            # Softmax
            y_hat = np.exp(o) / np.sum(np.exp(o), axis=0) 

            # Computing loss
            loss = -np.sum(y * np.log(y_hat))
            epoch_train_losses.append(loss)

            # Backward pass
            d_o = y_hat - y
            d_beta = np.dot(d_o, a_bias.T)
            # Excluding bias weight
            d_a = np.dot(beta[:, :-1].T, d_o)  
            d_z = d_a * a * (1 - a)
            d_alpha = np.dot(d_z, x)

            # Updating weights
            beta -= learning_rate * d_beta
            alpha -= learning_rate * d_alpha

        # Calculate mean loss for the epoch
        train_entropy.append(np.mean(epoch_train_losses))

        # Validation phase
        epoch_valid_losses = []
        for i in range(valid_x.shape[0]):
            
            x = valid_x[i, :][None, :]

            y = np.zeros((num_classes, 1))
            
            # One-hot encoding
            y[valid_y[i], 0] = 1 

            # Forward pass
            z = np.dot(alpha, x.T)
            a = 1 / (1 + np.exp(-z))
            a_bias = np.vstack((a, np.ones((1, 1)))) 
            o = np.dot(beta, a_bias)
            y_hat = np.exp(o) / np.sum(np.exp(o), axis=0)

            # Computing loss
            loss = -np.sum(y * np.log(y_hat))
            epoch_valid_losses.append(loss)

        # Calculating mean validation loss for the epoch
        valid_entropy.append(np.mean(epoch_valid_losses))

    return alpha, beta, train_entropy, valid_entropy

# Initializing the network parameters
def initialize_network(input_size, hidden_units, output_units, init_flag):
    alpha = np.random.uniform(-0.1, 0.1, (hidden_units, input_size + 1)) if init_flag else np.zeros((hidden_units, input_size + 1))
    beta = np.random.uniform(-0.1, 0.1, (output_units, hidden_units + 1)) if init_flag else np.zeros((output_units, hidden_units + 1))
    return alpha, beta

# One-hot encoding the labels
def one_hot_encode(label, num_classes):
    y_hot = np.zeros((num_classes, 1))
    y_hot[label] = 1
    return y_hot

def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param valid_x: Validation data input (size N_valid x M)
    :param valid_y: Validation labels (size N-valid x 1)
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    def forward_pass(x, alpha, beta):
        
        x_with_bias = np.hstack([np.ones((x.shape[0], 1)), x])
        # First linear step
        a = np.dot(x_with_bias, alpha.T)
        # Sigmoid activation
        z = 1 / (1 + np.exp(-a))
        # Add bias to hidden layer
        z_with_bias = np.hstack([np.ones((z.shape[0], 1)), z])
        # Second linear step
        b = np.dot(z_with_bias, beta.T)
        # Softmax
        y_hat = np.exp(b) / np.sum(np.exp(b), axis=1, keepdims=True)
        return y_hat
    
    # Forward pass for training and validation sets
    y_hat_train = forward_pass(tr_x, tr_alpha, tr_beta)
    y_hat_valid = forward_pass(valid_x, tr_alpha, tr_beta)

    # Convert probabilities to predicted labels
    predictions_train = np.argmax(y_hat_train, axis=1)
    predictions_valid = np.argmax(y_hat_valid, axis=1)

    # Calculate error rates
    train_error = np.mean(predictions_train != tr_y)
    valid_error = np.mean(predictions_valid != valid_y)

    return train_error, valid_error, predictions_train, predictions_valid


### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def initialize_params(n_in, n_out, init_rand):
    """Initializes weights and biases based on the given flag."""
    if init_rand:
        W = np.random.uniform(-0.1, 0.1, (n_out, n_in))
    else:
        W = np.zeros((n_out, n_in))
    b = np.zeros(n_out)
    return W, b

def forward_backward_pass(x, y, W1, b1, W2, b2, num_classes):
    """Performs a forward and backward pass through the network."""
    # Forward pass
    z1 = np.dot(W1, x) + b1
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation
    z2 = np.dot(W2, a1) + b2
    a2 = np.exp(z2 - np.max(z2))  # Softmax
    y_hat = a2 / np.sum(a2, axis=0, keepdims=True)
    
    # Backward pass
    y_true = np.zeros(num_classes)
    y_true[y] = 1
    d2 = y_hat - y_true
    dW2 = np.outer(d2, a1)
    db2 = d2
    da1 = np.dot(W2.T, d2)
    dz1 = da1 * a1 * (1 - a1)
    dW1 = np.outer(dz1, x)
    db1 = dz1

    return dW1, db1, dW2, db2, y_hat

def update_params(X, y, W1, b1, W2, b2, learning_rate, num_classes):
    """Updates the network parameters for one epoch."""
    for i in range(X.shape[0]):
        dW1, db1, dW2, db2, _ = forward_backward_pass(X[i], y[i], W1, b1, W2, b2, num_classes)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

def update_metrics(X, y, W1, b1, W2, b2, num_classes):
    """Computes and returns metrics for the dataset."""
    loss = 0
    correct_preds = 0
    predictions = []

    for i in range(X.shape[0]):
        _, _, _, _, y_hat = forward_backward_pass(X[i], y[i], W1, b1, W2, b2, num_classes)
        predictions.append(np.argmax(y_hat))
        loss += -np.log(y_hat[y[i]])
        correct_preds += int(y[i] == predictions[-1])

    avg_loss = loss / X.shape[0]
    error_rate = 1 - (correct_preds / X.shape[0])
    return avg_loss, predictions, error_rate

def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ Main function to train and validate your neural network implementation.

        X_train: Training input in N_train-x-M numpy nd array. Each value is binary, in {0,1}.
        y_train: Training labels in N_train-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        X_val: Validation input in N_val-x-M numpy nd array. Each value is binary, in {0,1}.
        y_val: Validation labels in N_val-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        num_hidden: Positive integer representing the number of hidden units.
        init_flag: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
        learning_rate: Float value specifying the learning rate for SGD.

        RETURNS: a tuple of the following six objects, in order:
        loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        y_hat_train: A list of integers representing the predicted labels for training data
        y_hat_val: A list of integers representing the predicted labels for validation data
    """    
    
    N_train, M = X_train.shape
    num_classes = len(np.unique(np.concatenate((y_train, y_val))))
    
    # Initialize weights and biases
    W1, b1 = initialize_params(M, num_hidden, init_rand)
    W2, b2 = initialize_params(num_hidden, num_classes, init_rand)
    
    #metrics
    metrics = {
        'loss_per_epoch_train': [],
        'loss_per_epoch_val': [],
        'err_train': [],
        'err_val': []
    }

    for epoch in range(num_epoch):
        # Updating weights and compute metrics for training data
        update_params(X_train, y_train, W1, b1, W2, b2, learning_rate, num_classes)
        train_loss, train_predictions, train_error = update_metrics(X_train, y_train, W1, b1, W2, b2, num_classes)
        metrics['loss_per_epoch_train'].append(train_loss)
        metrics['err_train'].append(train_error)

        # Computing metrics for validation data
        val_loss, val_predictions, val_error = update_metrics(X_val, y_val, W1, b1, W2, b2, num_classes)
        metrics['loss_per_epoch_val'].append(val_loss)
        metrics['err_val'].append(val_error)

    metrics['y_hat_train'] = train_predictions if epoch == num_epoch - 1 else []
    metrics['y_hat_val'] = val_predictions if epoch == num_epoch - 1 else []

    return (
        metrics['loss_per_epoch_train'],
        metrics['loss_per_epoch_val'],
        metrics['err_train'][-1] if metrics['err_train'] else None,
        metrics['err_val'][-1] if metrics['err_val'] else None,
        metrics['y_hat_train'],
        metrics['y_hat_val']
    )

