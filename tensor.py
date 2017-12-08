import tensorflow as tf

w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

x = tf.placeholder(tf.float32)

linear_model = w * x + b

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4]}))

#calculate Loss

y = tf.placeholder(tf.float32)
squaredata = tf.square(linear_model-y)
loss = tf.reduce_sum(squaredata)
print loss,squaredata
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

#Minimize the cost through gradient Descent

optimize = tf.train.GradientDescentOptimizer(0.01)
train = optimize.minimize(loss)

for i in range(1000):
    sess.run(train,{x :[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([w,b]))


