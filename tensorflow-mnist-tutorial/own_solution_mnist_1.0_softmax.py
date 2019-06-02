import tensorflow as tf
import mnistdata


def main():
    print("Tensorflow version " + tf.__version__)
    tf.set_random_seed(0)

    mnist = mnistdata.read_data_sets("data",
                                     one_hot=True,
                                     reshape=False)

    height = 28
    width = 28
    channels = 1
    output = 10
    minibatch_ph = None  # Will be determined during runtime
    minibatch = 100

    X = tf.placeholder(tf.float32, [minibatch_ph,
                                    height,
                                    width,
                                    channels])
    # NOTE: Y_ are correct answers
    Y_ = tf.placeholder(tf.float32, [minibatch_ph, output])
    W = tf.Variable(tf.zeros([height*width, output]))
    b = tf.Variable(tf.zeros([output]))

    # NOTE: Flatten
    # -1 means: Use the dimension which fits
    # In this case it will be minibatch*width
    XX = tf.reshape(X, [-1, height*width])
    Y = tf.nn.softmax(tf.matmul(XX, W) + b)

    # cross-entropy = - sum( Y_i * log(Yi) )
    # NOTE: We use mean to make things independent of batch size
    # NOTE: This is value is connected to batch size and learning rate
    #       In the original solution this was multiplied with 1000
    #       (100 from the batch size, 10 from number of outputs),
    #       and the learning rate was reduced by 1000
    cross_entropy = - tf.reduce_mean(Y_ * tf.log(Y))
    train_step = \
        tf.train.GradientDescentOptimizer(5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(2000 + 1):
        # Get the batches
        batch_X, batch_Y = mnist.train.next_batch(minibatch)

        # Training
        if i % 50 == 0:
            a, c = sess.run(
                [accuracy, cross_entropy],
                feed_dict={X: batch_X, Y_: batch_Y})

            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

        # Test values
        if i % 10 == 0:
            a, c = sess.run([accuracy, cross_entropy],
                             feed_dict={X: mnist.test.images,
                                        Y_: mnist.test.labels})

            print(str(i) + ": ********* epoch " + str(
                i * minibatch // mnist.train.images.shape[
                    0] + 1) + " ********* test accuracy:" + str(
                a) + " test loss: " + str(c))

        # Backprop
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})


if __name__ == '__main__':
    main()
