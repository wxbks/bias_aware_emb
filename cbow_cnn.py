#CBOW model version
from editor_article_matrix import processSentWiki
import string
import collections
import math
import os
import random
import zipfile
import json

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# pre-trained model file
pretrained_filename = 'glove.6B.100d.txt'

def gen_gloveVocab(pretrainModelTxt_):
    vocab = []
    embd = []
    pfile = open(pretrainModelTxt_, 'r')
    for line in pfile.readlines():
      row = line.strip().split()
      vocab.append(row[0])
      embd.append(row[1:])
    embeddings = np.asarray(embd)
    return vocab, embeddings

glove_vocab, gloveEmb = gen_gloveVocab(pretrained_filename)
print 'glove vocab size:',len(glove_vocab)


# enlarge 400000 glove emb to 419999 emb
# np.random.rand(19999,100)

# filename = '../classWordExtract/processedData/health_tweetId_processedTxt_label_health1Unrelated0.json'
# filename = '/home/sik211/dusk/SSWE/healthCareExtend/cbowM/cnn-text-classification-tf-master/data/health_related/train_health_processTxtALL.txt'
# filename = "/home/sik211/dusk/SSWE/healthCareExtend/classWordExtract/processedData/train_relatedVsNotRelated_ALL.txt"
filename = "wiki_trainEval_aclwrongRightTxtCol1011prepress.txt"


# Read the data into a list of strings.
# def read_data(filename):
  # """Extract the first file enclosed in a zip file as a list of words"""
  # with zipfile.ZipFile(filename) as f:
    # data = f.read(f.namelist()[0]).split()
  # return data

# def read_data(filename): 
  # with open(filename) as a:
    # f = json.load(a)
  # data = []
  # for l in f:
      # t = l[1].strip().split()
      # data = data + t
  # return data

# add pre-process to sents
def read_data(filename):
  t = list(open(filename, "r").readlines())
  t = [s.strip() for s in t]
  data = []
  for i in t:
    i = processSentWiki(i)
    isp = i.split()
    isp = [p.strip(string.punctuation) for p in isp]
    isp = [p.lower() for p in isp]
    data = data + isp
  return data

# words = read_data(filename)
# with open('cbow_cnn_words.json','w') as a:
    # json.dump(words,a)
with open('cbow_cnn_words.json') as b:
    words = json.load(b)

# convert words to lower cases
# for item in words:
    # item.lower()
        
print('self Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words, min_cut_freq):
  count_org = [['UNK', -1]]
  count_org.extend(collections.Counter(words).most_common())
  count = [['UNK', -1]]
  for word, c in count_org:
    word_tuple = [word, c]
    if word == 'UNK': 
        count[0][1] = c
        continue
    if c > min_cut_freq:
        count.append(word_tuple)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

min_cut_freq = 3 #cut off frequence smaller then 3 words
data, count, dictionary, reverse_dictionary = build_dataset(words, min_cut_freq)
print "done build dataset"
# enlarge vocabulary glove_vocab + self_vocab
self_vocab = reverse_dictionary.keys()
self_max = max([int(i) for i in self_vocab])
print 'come here after build_dataset'
# combine glove vocab to self vocab
# for w in glove_vocab:
  # if w not in reverse_dictionary.values():
      # self_max = self_max + 1
      # reverse_dictionary[str(self_max)] = w

newwords = set(glove_vocab).difference(set(reverse_dictionary.values()))
newwords = list(newwords)
for i in newwords:
    self_max = self_max + 1
    reverse_dictionary[self_max] = i
# enlarge 400000 glove emb to 4xxxxx emb
extra = len(reverse_dictionary) - len(glove_vocab)
extra_emb = np.random.rand(extra,100)
gloveEmb = np.concatenate((gloveEmb,extra_emb), axis=0)
gloveEmb.astype(float)
vocabulary_size = len(reverse_dictionary)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:20], [reverse_dictionary[i] for i in data[:20]])
print('Vocab size: ', vocabulary_size)
print('gloveEmb with extra shape: ', gloveEmb.shape)
# load chi2 stats to dict
# chi2_file = "../classWordExtract/processedData/health_chi2_dict.json"
# chi2_file = "/home/sik211/dusk/SSWE/healthCareExtend/classWordExtract/processedData/RelatedVsNotRelated20092012_chi2_dict.json"
chi2_file = "wiki_dict_chi2_biasedWord_20groupWithBiasedGroup.json"
chi2_dict = {}
with open(chi2_file) as a:
  chi2_dict = json.load(a)
chi2_ave = sum(chi2_dict.values())/float(len(chi2_dict.values()))
print("Chi2 stats loaded")

data_index = 0

def generate_cbow_batch(batch_size, num_skips, skip_window):
  global data_index
  global chi2_ave
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  chi2s = np.ndarray(shape=(batch_size, num_skips, 1), dtype=np.float32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    batch_temp = np.ndarray(shape=(num_skips), dtype=np.int32)
    chi2_temp = np.ndarray(shape=(num_skips, 1), dtype=np.float32)
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch_temp[j] = buffer[target]
      w = reverse_dictionary[buffer[target]]
      if w in chi2_dict:
        chi2_temp[j] = [ chi2_dict[w] ]
      else:
        chi2_temp[j] = [ chi2_ave ]
      chi2_temp = chi2_temp / np.sum(chi2_temp)
    batch[i] = batch_temp
    chi2s[i] = chi2_temp
          
    
    labels[i,0] = buffer[skip_window]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels, chi2s

# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 100  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

print valid_examples

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.float32,shape=[batch_size, skip_window * 2])
  train_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])
  train_chi2 = tf.placeholder(tf.float32, shape=[batch_size, skip_window * 2, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  # embedding_placeholder = tf.placeholder(tf.float32, [None, embedding_size], name='wtf')

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs
    # embeddings = tf.Variable(
        # tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    W = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, embedding_size]), trainable=True, name="W")
    
    embeddings = W.assign(gloveEmb)
    # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]
    train_inputs = tf.cast(train_inputs, tf.int32)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # print "embed:"
    # print embed
    embed = np.multiply(embed, train_chi2)
    # reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window*2)
    reduced_embed = tf.reduce_sum(embed, 1)
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, train_labels, reduced_embed,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul( valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 200001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")
  #session.run(embed)
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels, batch_chi2s = generate_cbow_batch(
        batch_size, num_skips, skip_window)
    # print gloveEmb.shape
    # gloveEmb = np.asarray(gloveEmb, np.float32)
    # gloveEmb = tf.convert_to_tensor(gloveEmb, np.float32)
    # print tf.shape(gloveEmb)
    # gloveEmb = gloveEmb.tolist()
    # gloveEmb = [map(float, i) for i in gloveEmb]
    # print batch_inputs.shape
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels, train_chi2 : batch_chi2s}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
    # print(embeds.shape)
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0
                
    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 100000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


# np.save("health_chi2_weighted_cbowEmbeddings.npy", final_embeddings)
# with open("health_chi2_weighted_cbowDict_KeyrowNum_ValWord.json", "w") as a:
  # json.dump(reverse_dictionary, a)

# np.save("relatedVsNotRelated_chi2_weighted_cbowEmbeddings.npy", final_embeddings)
# with open("relatedVsNotRelated_chi2_weighted_cbowDict_KeyrowNum_ValWord.json", "w") as a:
  # json.dump(reverse_dictionary, a)
np.save("wiki_trainEval_mixedWrongRight_emb100_glovePre_step200000", final_embeddings)
with open("wiki_trainEval_mixedWrongRight_emb100_chi2_weighted_cbowDict_KeyrowNumValWord_glove_step200000.json","w") as a:
  json.dump(reverse_dictionary, a)

tsne_plot = False # Change the false to true to invoke the tsne graph

# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne_CBOW.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)
  
try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  if(tsne_plot):
      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 300
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")
# Testing final embedding
input_dictionary = dict([(v,k) for (k,v) in reverse_dictionary.iteritems()])

test_word_idx_a = input_dictionary.get('france') # replace france with other testing words such as king
test_word_idx_b = input_dictionary.get('paris')    # replace paris with other testing words such as man
test_word_idx_c = input_dictionary.get('rome')    # replace rome with other testing words such as woman

a = final_embeddings[test_word_idx_a,:]
b = final_embeddings[test_word_idx_b,:]
c = final_embeddings[test_word_idx_c,:]
ans = c + (b - a)

similarity = final_embeddings.dot(ans)
top_k = 4
nearest = (-similarity).argsort()[0:top_k+1]
print nearest
for k in xrange(top_k+1):
    close_word = reverse_dictionary[nearest[k]]
    print(close_word)

