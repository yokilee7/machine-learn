import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# function add layer
def add_layer(inputs, in_size, out_size, activation_function = None):
	# define weights and biases as a tensorflow variable and get a random initialize
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

	# define the calculation
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	# us the activation_function
	if activation_function is None :
		outputs = Wx_plus_b
	else :
		outputs = activation_function(Wx_plus_b)
	return outputs

# make x dataset
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
# calculate y dataset
y_data = np.square(x_data) - 0.5 + noise

# define placeholder
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# hide layer 
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
# out layer
prediction = add_layer(l1, 10, 1, activation_function = None)

# calculate loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))

# training step
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

# data graphically
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# training
for i in range(1000):
	sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
	if i % 50:
		#print(sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:x_data})
		lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
		plt.pause(0.1)