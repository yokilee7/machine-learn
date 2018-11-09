# http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
import tensorflow as tf
import input_data


# read training data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

# creat a interactive session
sess = tf.InteractiveSession()

#creat the calculation map
x = tf.placeholder('float', shape = [None, 784])
y_ = tf.placeholder('float', shape = [None, 10])