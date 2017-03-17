import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

nb_classes = 43
epochs = 10
batch_size = 128

with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and validation sets.
x_train, x_val, y_train, y_val = train_test_split(data['features'],
                                                  data['labels'], test_size=.33, random_state=0)

# Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, [227, 227])

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
init_op = tf.global_variables_initializer()
predictions = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def evaluate(x_data, y_data, sess):
    x_data, y_data = shuffle(x_data, y_data)

    total_accuracy = 0
    total_loss = 0
    for offset in range(0, x_data.shape[0], batch_size):

        batch_end = offset + batch_size
        batch_x = x_data[offset:batch_end]
        batch_y = y_data[offset:batch_end]

        loss, accuracy = sess.run([loss_operation, accuracy_operation],
                                  feed_dict={features: batch_x, labels: batch_y})

        total_loss += (loss * batch_x.shape[0])
        total_accuracy += (accuracy * batch_x.shape[0])

    return total_loss/x_data.shape[0], total_accuracy/x_data.shape[0]


# Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):

        x_train, y_train = shuffle(x_train, y_train)

        sample_count = x_train.shape[0]

        start = time.time()

        for offset in range(0, sample_count, batch_size):

            batch_end = offset + batch_size

            sess.run(training_operation, feed_dict={features: x_train[offset:batch_end],
                                                    labels: y_train[offset:batch_end]})

        validation_loss, validation_acc = evaluate(x_val, y_val, sess)
        print("Epoch", i)
        print("Time: %.3f seconds" % (time.time() - start))
        print("Validation loss: ", validation_loss)
        print("Validation accuracy: ", validation_acc)
        print()
