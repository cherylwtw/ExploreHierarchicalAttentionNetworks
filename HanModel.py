import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, vocab_size, num_classes, embedding_size=200, hidden_size=50):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

        word_embedded = self.word2vec()
        sent_vec = self.sent2vec(word_embedded)
        doc_vec = self.doc2vec(sent_vec)
        out = self.classifer(doc_vec)

        self.out = out


    def word2vec(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        return word_embedded

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output