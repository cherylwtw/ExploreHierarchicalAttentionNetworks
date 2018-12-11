import tensorflow as tf
import time
import os
from ReadData import get_test_data
from HanModel import HAN
import sys
import os
from sklearn.model_selection import train_test_split


num_per_class = 20000
max_sent = 15
max_word = 15

# Define some variables
vocab_size = 46960
num_classes = 5
embedding_size = 200
hidden_size = 50
batch_size = 32
num_epochs = 10
num_checkpoints = 5
learning_rate = 0.01
grad_clip = 5

test_x, test_y = get_test_data(num_per_class, max_sent, max_word)
test_x, _, test_y, _ = train_test_split(test_x, test_y, test_size=0, random_state=42)
test_set_size = len(test_y)

print("loaded testing data")
print(test_x.shape)
print(test_y.shape)

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
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp + '_test'))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    sess.run(tf.global_variables_initializer())


    def predict_step(x_test, y_test, model_num, writer=False):
        cost_list = []
        acc_list = []
        for j in range(0, test_set_size, batch_size):
            x_valid_batch = x_test[j:j + batch_size]
            y_valid_batch = y_test[j:j + batch_size]

            num_in_batch = len(y_valid_batch)

            feed_dict = {
                han.input_x: x_valid_batch,
                han.input_y: y_valid_batch,
                han.max_sentence_num: 15,
                han.max_sentence_length: 15,
                han.batch_size: 64
            }

            # step, cost, accuracy = sess.run([global_step, loss, acc], feed_dict)
            cost, accuracy = sess.run([loss, acc], feed_dict)
            weighted_cost = cost * num_in_batch
            weighted_acc = accuracy * num_in_batch

            cost_list.append(weighted_cost)
            acc_list.append(weighted_acc)
        cost_avg = sum(cost_list) / len(y_test)
        acc_avg = sum(acc_list) / len(y_test)

        time_str = str(int(time.time()))
        print("++++++++++++++++++valid++++++++++++++{}:loss {:g}, acc {:g}".format(time_str, cost_avg, acc_avg))

        # loss_avg = tf.summary.scalar('loss', cost_avg)
        # acc_avg = tf.summary.scalar('accuracy', acc_avg)
        # dev_summary_op = tf.summary.merge([loss_avg, acc_avg])
        # summaries = sess.run([dev_summary_op])
        # dev_summary_writer.add_summary(summaries, step)
        if writer:
            f = open(out_dir + "/result.txt", "w+")
            f.write("Model: %s\r\n" % model_num)
            f.write("accuracy: %f\r\n" % accuracy)
            f.close()


    def main(model_num):
        # if is_continuous is true, then do model restore
        in_model_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_num, "checkpoints"))
        saver.restore(sess, in_model_dir + "/model.ckpt")

        predict_step(test_x, test_y,model_num, writer=True)

    if __name__ == '__main__':
        try:
            arguments = sys.argv[1:]
            is_model = '-model' in arguments
            model_num = ''
            if is_model:
                model_idx = arguments.index('-model')
                model_num = arguments[model_idx + 1]
        except:
            print("print usage")

        main(model_num)
