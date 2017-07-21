from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from PIL import Image

class MnistRegressTest(object):
    def __init__(self):
        mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
        self.mnist = mnist


    # def init_data(self):
    #     mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
    #     self.mnist = mnist

    def regress(self):
        max_step = 1000
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
                # if step % 100 == 0:
                #     print(sess.run(W))
                #     print(sess.run(b))
                #     print("----------------")
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            print("exit train....")

            #test data
            pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            acc = tf.reduce_mean(tf.cast(pred, tf.float32))
            print(sess.run(acc, feed_dict={x: self.mnist.test.images, y_: self.mnist.test.labels}))
            print(self.mnist.test.labels)


    def regressForInteractive(self):
        sess = tf.InteractiveSession()

        x = tf.placeholder("float", shape=[None, 784])
        y_ = tf.placeholder("float", shape=[None, 10])
        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))
        sess.run(tf.global_variables_initializer())

        y = tf.nn.softmax(tf.matmul(x,W) + b)
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        for i in range(1000):
          batch = self.mnist.train.next_batch(50)
          train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={x: self.mnist.test.images, y_: self.mnist.test.labels}))
        sess.close()





    def covertImage(self):
        newImage = Image.new('L', (28, 28)) #转化成黑白的（L表示黑白模式）
        a = self.mnist.test.images[1].reshape([28, 28])
        # print(a)
        a = a * 255
        # print(a)
        for y in range(28):
            for x in range(28):
                newImage.putpixel((x, y), int(a[x][y]))

        # img = Image.fromarray(a)
        # if img.mode != 'L':
        #     img = img.convert('L')
        newImage.save("3.png")

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        pic = self.mnist.test.images[1].reshape([28, 28])
        plt.imshow(pic,cmap = cm.binary)
        plt.show()




if __name__ == "__main__":
    print("test begin...")
    testInst = MnistRegressTest()
    testInst.covertImage()
    testInst.regress()
    testInst.regressForInteractive()

    print("test end...")