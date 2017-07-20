from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class MnistTest(object):
    def __init__(self):
        mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
        self.mnist = mnist


    # def init_data(self):
    #     mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
    #     self.mnist = mnist

    def regress(self):
        max_step = 10000
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10], name='W'))
        b = tf.Variable(tf.zeros([10]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder(tf.float32, [None, 10])

        # lrn_rate = tf.constant(0.01)
        lrn_rate = 0.01
        batch_size = 100
        init_op = tf.global_variables_initializer()

        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        # cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

        train_step = tf.train.GradientDescentOptimizer(lrn_rate).minimize(cross_entropy)

        with tf.Session() as sess:
            print("begin train...")

            sess.run(init_op)
            for step in range(max_step):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                if step % 100 == 0:
                    print(sess.run(W))
                    print(sess.run(b))
                    print("----------------")
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            print("exit train....")

            #test data
            pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            acc = tf.reduce_mean(tf.cast(pred, tf.float32))
            print(sess.run(acc, feed_dict={x: self.mnist.test.images, y_: self.mnist.test.labels}))
            print(self.mnist.test.labels)






def testReshape():
    import numpy as np
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    b = a.reshape([-1, 2])
    c = a.reshape([1, 2, 4])
    print(b)
    print(c)

if __name__ == "__main__":
    print("test begin...")
    # testShape()
    testInst = MnistTest()
    testInst.regress()

    print("test end...")