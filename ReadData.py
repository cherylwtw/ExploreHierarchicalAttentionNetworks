from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pickle
import numpy as np
import sys
from termcolor import colored

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

def add_vocab_count(df):
    try:
        vocab_file = open("vocab", 'rb')
        vocab_old = pickle.load(vocab_file)
        print (colored('load vocab done', 'red'))
        print_dict_by_key_order(vocab_old)
    except FileNotFoundError:
        vocab_old = {}

    df['tokenized_sents'] = df.apply(lambda row: word_tokenize(row['reviewText']), axis=1)
    print("done tokenized sentence")
    print("df's shape is " + str(df.shape))
    print("print out the first 5 rows as example")
    print(df.head(5))
    df['vocab_dict'] = df.apply(lambda row: Counter(row['tokenized_sents']), axis=1)
    print("done counting")
    print("df's shape is " + str(df.shape))
    print("print out the first 5 rows as example")
    print(df.head(5))
    vocab_dict_list = df['vocab_dict'].tolist()
    vocab_dict_new_sum = sum((Counter(dict(x)) for x in vocab_dict_list), Counter())
    print(colored('done summing: dictionary to add'), 'red')
    print_dict_by_key_order(vocab_dict_new_sum)

    vocab_list = [vocab_old, vocab_dict_new_sum]

    vocab_dict_sum_final = sum((Counter(dict(x)) for x in vocab_list), Counter())

    with open("vocab", 'wb') as g:
        print(colored('add to vocab:', 'red'))
        print_dict_by_key_order(vocab_dict_new_sum)
        pickle.dump(vocab_dict_sum_final, g)
        print("vocab saved")


def print_dict_by_key_order(dict):
    print(sorted(dict.items(), key=lambda kv: kv[1], reverse=True))

def get_random_sample(class_num, sample_per_class):
    print("get ramdom sample for class " + str(class_num) + ":")
    df = pd.read_csv('cheryl_amazon_data_' + str(class_num) + '.0.csv')
    # remove header rows
    df = df[df.overall != 'overall']
    df = df.dropna(subset=['reviewText'])
    # take random samples
    df = df.sample(n=sample_per_class)

    # save it to pickle file
    pickle.dump(df, open("cheryl_amazon_data_top" + str(sample_per_class) + "_" + str(class_num) + ".0", 'wb'))


def add_to_vocab(class_num, sample_per_class):
    # read each file individually to build vocab
    print("add class " + str(class_num) + " to vocab" + ":")
    truncated_review_file = open("cheryl_amazon_data_top" + str(sample_per_class) + "_" + str(class_num) + ".0", 'rb')
    df = pickle.load(truncated_review_file)
    add_vocab_count(df)

def word_freq_to_vocab():
    # get_vocab
    vocab = get_vocab()
    # remove words that has less than 5 counts, treat them as UNKNOWN
    vocab_greater_5 = {key: val for key, val in vocab.items() if val > 5}

    print(colored("word freq > 5", 'red'))
    print_dict_by_key_order(vocab_greater_5)
    pickle.dump(vocab_greater_5, open("word_freq_greater_5", 'wb'))

    print("final vocab")
    final_vocab = {}
    final_vocab['UNKNOWN'] = 0
    id = 0
    for word, freq in vocab_greater_5.items():
        final_vocab[word] = id
        id += 1

    print_dict_by_key_order(final_vocab)
    pickle.dump(final_vocab, open("final_vocab", 'wb'))


def build_dataset(sample_per_class, class_num, max_sent_in_doc, max_word_in_sent):
    print(colored("build dataset for " + str(sample_per_class) + "_" + str(class_num)), 'red')
    # get_vocab
    final_vocab_file = open("final_vocab", 'rb')
    vocab = pickle.load(final_vocab_file)

    print("loaded final vocab")
    print_dict_by_key_order(vocab)

    num_class = 5

    truncated_review_file = open("cheryl_amazon_data_top" + str(sample_per_class) + "_" + str(class_num) + ".0", 'rb')
    df = pickle.load(truncated_review_file)
    print(colored("loaded reviews ", 'red'))
    print(colored(df.shape, 'red'))

    review_num = df.shape[0]
    df = df.reset_index(drop=True)

    data_x = np.zeros([review_num, max_sent_in_doc, max_word_in_sent])
    data_y = []

    for review_idx, review in df.iterrows():
        if review_idx % 1000 == 0:
            print("processing review # " + str(review_idx))
        try:
            star = int(float(review['overall']))
            sents = sent_tokenize(review['reviewText'])
        except:
            print("skip row " + str(review_idx))
            continue

        review_data = np.zeros([max_sent_in_doc, max_word_in_sent])

        for i, sent in enumerate(sents):
            if i < max_sent_in_doc:
                sent_data = np.zeros([max_word_in_sent], dtype=int)
                for j, word in enumerate(word_tokenize(sent)):
                    if j < max_word_in_sent:
                        sent_data[j] = vocab.get(word, 0)
                review_data[i] = sent_data

        data_x[review_idx] = review_data
        labels = [0] * num_class
        labels[star - 1] = 1
        data_y.append(labels)

    pickle.dump((data_x, data_y), open("cheryl_amazon_data_processed_" + str(class_num) + ".0", 'wb'))
    print("done saving data_x and data_y")

def get_vocab():
    try:
        vocab_file = open("vocab", 'rb')
        vocab = pickle.load(vocab_file)
        print ("load vocab done")
    except FileNotFoundError:
        vocab = {}
    return vocab

def get_data(class_num):
    try:
        data_file = open("cheryl_amazon_data_processed_" + str(class_num) + ".0", 'rb')
        data_x, data_y = pickle.load(data_file)
        print ("load data done")
    except FileNotFoundError:
        data_x = []
        data_y = []
    return data_x, data_y


def prepare_separate_data():
    data_x_1, data_y_1 = get_data(1)
    data_x_2, data_y_2 = get_data(2)
    data_x_3, data_y_3 = get_data(3)
    data_x_4, data_y_4 = get_data(4)
    data_x_5, data_y_5 = get_data(5)

    # split train and validation
    train_x_1, valid_x_1, train_y_1, valid_y_1 = train_test_split(data_x_1, data_y_1, test_size=0.10, random_state=1)
    train_x_2, valid_x_2, train_y_2, valid_y_2 = train_test_split(data_x_2, data_y_2, test_size=0.10, random_state=2)
    train_x_3, valid_x_3, train_y_3, valid_y_3 = train_test_split(data_x_3, data_y_3, test_size=0.10, random_state=3)
    train_x_4, valid_x_4, train_y_4, valid_y_4 = train_test_split(data_x_4, data_y_4, test_size=0.10, random_state=4)
    train_x_5, valid_x_5, train_y_5, valid_y_5 = train_test_split(data_x_5, data_y_5, test_size=0.10, random_state=4)

    train_full_x = np.concatenate((train_x_1, train_x_2, train_x_3, train_x_4, train_x_5), axis=0)
    train_full_y = np.concatenate((train_y_1, train_y_2, train_y_3, train_y_4, train_y_5), axis=0)

    valid_full_x = np.concatenate((valid_x_1, valid_x_2, valid_x_3, valid_x_4, valid_x_5), axis=0)
    valid_full_y = np.concatenate((valid_y_1, valid_y_2, valid_y_3, valid_y_4, valid_y_5), axis=0)

    pickle.dump((train_full_x, train_full_y), open("train_data", 'wb'))
    pickle.dump((valid_full_x, valid_full_y), open("valid_data", 'wb'))

    print("full training set size: ")
    print(train_full_x.shape)
    print(train_full_y.shape)
    print("full validation set size: ")
    print(valid_full_x.shape)
    print(valid_full_y.shape)


def prepare_test_data():
    data_x_1, data_y_1 = get_data(1)
    data_x_2, data_y_2 = get_data(2)
    data_x_3, data_y_3 = get_data(3)
    data_x_4, data_y_4 = get_data(4)
    data_x_5, data_y_5 = get_data(5)

    test_full_x = np.concatenate((data_x_1, data_x_2, data_x_3, data_x_4, data_x_5), axis=0)
    test_full_y = np.concatenate((data_y_1, data_y_2, data_y_3, data_y_4, data_y_5), axis=0)

    pickle.dump((test_full_x, test_full_y), open("test_data", 'wb'))

    print("full testing set size: ")
    print(test_full_x.shape)
    print(test_full_y.shape)


def get_train_data(num_per_class, max_sent, max_word):
    dir_name = "data_" + str(num_per_class) + "_" + str(max_sent) + "_" + str(max_word)
    data_file = open(dir_name + "/train_data", 'rb')
    train_x, train_y = pickle.load(data_file)
    return train_x, train_y

def get_valid_data(num_per_class, max_sent, max_word):
    dir_name = "data_" + str(num_per_class) + "_" + str(max_sent) + "_" + str(max_word)
    data_file = open(dir_name + "/valid_data", 'rb')
    valid_x, valid_y = pickle.load(data_file)
    return valid_x, valid_y

def get_test_data(num_per_class, max_sent, max_word):
    dir_name = "test_data_" + str(num_per_class) + "_" + str(max_sent) + "_" + str(max_word)
    data_file = open(dir_name + "/test_data", 'rb')
    test_x, test_y = pickle.load(data_file)
    return test_x, test_y

if __name__ == '__main__':
    num_per_class = 2000
    get_random_sample(1, num_per_class)
    get_random_sample(2, num_per_class)
    get_random_sample(3, num_per_class)
    get_random_sample(4, num_per_class)
    get_random_sample(5, num_per_class)
    
    add_to_vocab(1, num_per_class)
    add_to_vocab(2, num_per_class)
    add_to_vocab(3, num_per_class)
    add_to_vocab(4, num_per_class)
    add_to_vocab(5, num_per_class)

    word_freq_to_vocab()
   
    build_dataset(num_per_class, 1, 30, 30)
    build_dataset(num_per_class, 2, 30, 30)
    build_dataset(num_per_class, 3, 30, 30)
    build_dataset(num_per_class, 4, 30, 30)
    build_dataset(num_per_class, 5, 30, 30)
    # prepare_test_data()
    prepare_separate_data()

