import tensorflow as tf
import time
import os
from ReadData import get_train_data, get_valid_data, get_vocab
from HanModel import HAN
import sys
import os
import shutil
from sklearn.model_selection import train_test_split

num_per_class = 10000
max_sent = 30
max_word = 30
train_x, train_y = get_train_data(num_per_class, max_sent, max_word)
valid_x, valid_y = get_valid_data(num_per_class, max_sent, max_word)

# shuffle
train_x, _, train_y, _ = train_test_split(train_x, train_y, test_size=0, random_state=42)
valid_x, _, valid_y, _ = train_test_split(valid_x, valid_y, test_size=0, random_state=13)


print("load training data " + str(len(train_y)))
print("load valid data " + str(len(valid_y)))

print("training data")
print(train_x.shape)
print(train_y.shape)
print(train_x)
print(train_y)

print("validation data")
print(valid_x.shape)
print(valid_y.shape)
print(valid_x)
print(valid_y)

# Define some variables
vocab_size = 46960
num_classes = 5
embedding_size = 200
hidden_size = 50
batch_size = 32
num_checkpoints = 5
learning_rate = 0.01
grad_clip = 5
num_epochs = 10
training_set_size = len(train_y)
valid_set_size = len(valid_y)


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


with tf.Session() as sess:
    han = HAN(vocab_size=vocab_size,
                    num_classes=num_classes,
                    embedding_size=embedding_size,
                    hidden_size=hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=han.input_y,
                                                                      logits=han.out,
                                                                      name='loss'))
    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        label = tf.argmax(han.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)


    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    out_train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(out_train_summary_dir, sess.graph)

    out_dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(out_dev_summary_dir, sess.graph)

    out_model_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 30,
            han.max_sentence_length: 30,
            han.batch_size: 64
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)

        time_str = str(int(time.time()))
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)

        return step

    def valid_step(x_valid, y_vaild, batch_size,  writer=None):
        cost_list = []
        acc_list = []
        for j in range(0, valid_set_size, batch_size):
            x_valid_batch = x_valid[j:j + batch_size]
            y_valid_batch = y_vaild[j:j + batch_size]

            num_in_batch = len(y_valid_batch)

            feed_dict = {
                han.input_x: x_valid_batch,
                han.input_y: y_valid_batch,
                han.max_sentence_num: 30,
                han.max_sentence_length: 30,
                han.batch_size: 64
            }

            step, cost, accuracy = sess.run([global_step, loss, acc], feed_dict)
            weighted_cost = cost * num_in_batch
            weighted_acc = accuracy * num_in_batch

            cost_list.append(weighted_cost)
            acc_list.append(weighted_acc)
        cost_avg = sum(cost_list) / len(y_vaild)
        acc_avg = sum(acc_list) / len(y_vaild)

        time_str = str(int(time.time()))
        print("++++++++++++++++++valid++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str,
                                                                                                 step, cost_avg, acc_avg))

        if writer:
            summary = tf.Summary(value=[tf.Summary.Value(tag="loss_1", simple_value=cost_avg),
                                        tf.Summary.Value(tag="accuracy_1", simple_value=acc_avg)])
            writer.add_summary(summary, step)
            saver.save(sess, out_model_dir + "/model.ckpt")



    def main(epoch_num, is_continuous, continuous_from_run):
        # if is_continuous is true, then do model restore
        if is_continuous:
            in_model_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", continuous_from_run, "checkpoints"))
            saver.restore(sess, in_model_dir + "/model.ckpt")
        evaluate_every = 100
        checkpoint_every = 100
        for epoch in range(epoch_num):
            print('current epoch %s' % (epoch + 1))
            for i in range(0, training_set_size, batch_size):
                x = train_x[i:i + batch_size]
                y = train_y[i:i + batch_size]
                step = train_step(x, y)
                
                if step % 10000 == 0:
                    evaluate_every += 100
                    checkpoint_every += 100
                    print("evaluate_every updated to: " + str(evaluate_every))
                if step % evaluate_every == 0:
                    valid_step(valid_x, valid_y, batch_size, dev_summary_writer)

    if __name__ == '__main__':
        try:
            arguments = sys.argv[1:]
            is_epoch = '-epoch' in arguments
            if is_epoch:
                num_epochs_idx = arguments.index('-epoch')
                num_epochs = int(arguments[num_epochs_idx + 1])

            continuous_from = ''
            is_continuous = '-continuous' in arguments
            if is_continuous:
                continuous_idx = arguments.index('-continuous')
                continuous_from = arguments[continuous_idx + 1]
            # else it is for testing, we get the model number
        except:
            print("failed: usage")

        main(num_epochs, is_continuous, continuous_from)
