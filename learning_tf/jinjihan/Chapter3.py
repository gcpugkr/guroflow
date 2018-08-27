
# coding: utf-8

# In[2]:


import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a, b)
e = tf.add(c, b)
f = tf.subtract(d, e)

sess = tf.Session()
outs = sess.run(f)
sess.close()
print("outs = {}".format(outs))


# In[15]:


import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)

c = tf.multiply(a, b)
d = tf.add(a,b)
e = tf.subtract(c, d)
f = tf.add(c, d)
g = tf.divide(e, f)

sess = tf.Session()
outs = sess.run(g)
sess.close()
print("outs = {}".format(outs))


# In[23]:


import tensorflow as tf

a = tf.constant(0.5)
b = tf.constant(0.2)

c = tf.multiply(a, b)
d = tf.sin(c)
e = tf.divide(d, b)

sess = tf.Session()
outs = sess.run(e)
sess.close()
print("outs = {}".format(outs))


# In[26]:


import tensorflow as tf
print(tf.get_default_graph())

g = tf.Graph()
print(g)

a = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.get_default_graph())


# In[27]:


import tensorflow as tf

g1 = tf.get_default_graph()
g2 = tf.Graph()

print(g1 is tf.get_default_graph())

with g2.as_default():
    print(g1 is tf.get_default_graph())

print(g1 is tf.get_default_graph())


# In[28]:


import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a, b)
e = tf.add(c, b)
f = tf.subtract(d, e)

with tf.Session() as sess:
    fetches = [a,b,c,d,e,f]
    outs = sess.run(fetches)
    
print("outs = {}".format(outs))
print(type(outs[0]))


# In[32]:


c = tf.constant(4.0)
print(c)


# In[33]:


c = tf.constant(4.0, dtype=tf.float64)
print(c)
print(c.dtype)


# In[35]:


x = tf.constant([1,2,3],name='x',dtype=tf.float32)
print(x.dtype)
x = tf.cast(x, tf.int64)
print(x.dtype)


# In[37]:


import numpy as np

c = tf.constant([[1,2,3],
                [4,5,6]])
print("Python List input: {}".format(c.get_shape()))

c = tf.constant(np.array([
    [[1,2,3],
    [4,5,6]],
    [[1,1,1],
    [2,2,2]]
]))
print("3d Numpy array input: {}".format(c.get_shape()))


# In[38]:


sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0, 5)
print("The content of 'c':\n {}\n".format(c.eval()))
sess.close()


# In[47]:


Ax = b
A = tf.constant([[1,2,3],
                [4,5,6]])
print(A.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

x = tf.expand_dims(x,1)
print(x.get_shape())

b = tf.matmul(A,x)

sess = tf.InteractiveSession()
print('matual result:\n {}'.format(b.eval()))
sess.close()


# In[50]:


with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    c2 = tf.constant(4,dtype=tf.int32,name='c')

print(c1.name)
print(c2.name)


# In[51]:


with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    with tf.name_scope("prefx_name"):
        c2 = tf.constant(4,dtype=tf.int32,name='c')
        c3 = tf.constant(4,dtype=tf.float64,name='c')

print(c1.name)
print(c2.name)
print(c3.name)


# In[58]:


init_val = tf.random_normal((1,5),0,1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print("\npost run: \n{}".format(post_var))


# In[64]:


x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,shape=(5,10))
    w = tf.placeholder(tf.float32,shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw= tf.matmul(x,w)
    
    xwb = xw + b
    
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s,feed_dict={x: x_data,w: w_data})
    
    print("outs = {}".format(outs))


# In[84]:


import numpy as np

x_data = np.random.randn(2000, 3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

NUM_STEPS = 10

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b
        
    with tf.name_scope('lose') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
        
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
            
        print(10, sess.run([w,b]))


# In[83]:


N = 20000
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_data = np.random.randn(N,3)
w_real = [0.3,0.5,0.1]
b_real = 0.2
wxb = np.matmul(w_real, x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)

NUM_STEPS = 50


##########
g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b
        
    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)
        
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
            
        print(50, sess.run([w,b]))

