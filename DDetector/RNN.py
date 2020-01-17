from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from  Datakeeper import *
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split


# Training Parameters
learning_rate = 0.0001
training_steps = 10000
batch_size = 64
display_step = 5

# Network Parameters
num_input = 64
timesteps = 64
num_hidden = 64
num_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
accuracy_history = []

def sortFirt(val):
    return val[0]

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def configure_RNN():
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return   train_op ,accuracy , loss_op, prediction

def getPrefMat(test_y,pred , m):
    prefmat = m
    for line in zip(test_y,pred):
        pred_lie = False
        act_lie = False
        if line[1][0] > 0.5 : pred_lie = True
        if line[0][0] > 0.5 : act_lie = True
        if (act_lie and pred_lie):
            prefmat[1][1] = prefmat[1][1] +1
        if not act_lie and not pred_lie:
            prefmat[0][0] = prefmat[0][0] +1
        if (act_lie and not pred_lie):
            prefmat[1][0] = prefmat[1][0] +1
        if (not act_lie and pred_lie):
            prefmat[0][1] = prefmat[0][1] +1

    return prefmat

def run_RNN( train_op ,accuracy , loss_op, prediction, train_data ,test_data , training_epochs = 10):
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    test_acurecy = 0
    best_test_acureccy = 0
    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for epoc in range(1, training_epochs+1):
            total_batch = train_data.getNumOfBatches()
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_data.getNextBatch()
                print(batch_x.shape)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

                if epoc % display_step == 0 or epoc == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
                    print("epoc " + str(epoc) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))


            if epoc % display_step == 0 or epoc == 1:
                m = [[0,0],[0,0]]
                total_batch_test = test_data.getNumOfBatches()
                for i in range(total_batch_test):
                    test_x, test_y = test_data.getNextBatch()
                    best_test_acureccy = max(test_acurecy , best_test_acureccy)
                    test_acurecy, pred = sess.run([accuracy, prediction], feed_dict={X: test_x, Y: test_y})
                    m = getPrefMat(test_y,pred , m)
                #test_x = test_x.reshape((-1, timesteps, num_input))

                print("Epoc: " ,epoc , " Testing Accuracy:", test_acurecy)

                print(m)
                persition = m[1][1]/(m[1][1] + m[1][0])
                accuracy_history.append([test_acurecy , m , persition])
                #if test_acurecy > 0.64:
                #    print("Accuracy reched")
                #    break

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        #test_x, test_y = test_data.getNextBatch()
        #test_x = test_x.reshape((-1, timesteps, num_input))

        #print("Testing Accuracy:", \
        #    sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
        accuracy_history.sort(key = sortFirt , reverse= True)
        print("best Accuracy: ",accuracy_history[0])
        import copy
        tmp = copy.deepcopy(accuracy_history[0])
        accuracy_history.clear()
        return tmp



'''
def main():

    train_data  = DataKeeper(data_image_train,data_label_train, label_names )
    train_data.setBatchSize(batch_size)
    test_data   = DataKeeper(data_image_test,data_label_test, label_names )
    test_data.setBatchSize(batch_size)
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    test_acurecy = 0
    best_test_acureccy = 0
    overfit = False
    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for epoc in range(1, training_epochs+1):
            total_batch = train_data.getNumOfBatches()
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_data.getNextBatch()

                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if epoc % display_step == 0 or epoc == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("epoc " + str(epoc) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            # Calculate accuracy for 128 mnist test images
            if epoc % display_step == 0 or epoc == 1:
                test_len = 128
                test_x, test_y = test_data.getNextBatch()
                test_x = test_x.reshape((-1, timesteps, num_input))
                best_test_acureccy = max(test_acurecy , best_test_acureccy)
                test_acurecy = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
                print("Testing Accuracy:", test_acurecy)
                print("best Testing Accuracy:", best_test_acureccy)

                if best_test_acureccy > test_acurecy + 0.2:
                    print("Accuracy starting to decent stoping traning")
                    overfit = True
                    break
            if overfit:
                break

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_x, test_y = test_data.getNextBatch()
        test_x = test_x.reshape((-1, timesteps, num_input))

        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))

if __name__ == "__main__":
    main()

'''
