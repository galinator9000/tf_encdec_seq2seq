#! -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import re

from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors

# Applies filter to given list. Clears empty elements.
def apply_filter(x):
	return list(filter(None, x))

# Cleans given data.
def clean_data(data):
	data = re.sub(r"""[^A-Za-z0-9 \n\t]""", " ", data)

	for i in range(0, 100):
		data = data.replace("  ", " ")
	
	data = data.replace("\n ", "\n").replace(" \n", "\n").replace("\t ", "\t").replace(" \t", "\t").replace(" '", "'").replace("' ", "'")
	return data.lower()

# Pads given sentence with <pad> to nearest bucket.
def pad_to_bucket(x, bucket_structure, is_input, hParams):
	x_split = apply_filter(x.split(" "))

	bucketIndex = -1
	for b in range(0, len(bucket_structure)):
		bucket = bucket_structure[b]
		if len(x_split) <= bucket[0]:
			bucketIndex = b
			break
	if bucketIndex == -1:
		return x

	input_gap = bucket_structure[bucketIndex][0] - len(x_split)

	if is_input:
		return (str((hParams.vocab_special_token[2] + " ")*input_gap) + x).strip()
	else:
		return (x + str((hParams.vocab_special_token[2] + " ")*input_gap)).strip()

# Decodes given vector from vocabulary.
def decode_seq(x, l):
	sentence = ""
	for t in range(0, x.shape[0]):
		ind = np.argmax(x[t])
		sentence += l[ind] + " "
	sentence = sentence.strip()
	return sentence

# Decodes given vector from Embedding model.
def decode_seq_vector(x, model_embedding, hParams):
	sentence = ""
	for t in range(0, x.shape[0]):
		try:
			mostSimilar = model_embedding.wv.similar_by_vector(x[t])[0]
		except IndexError:
			sentence += hParams.vocab_special_token[3] + " "
			continue

		sentence += mostSimilar[0] + " "
	sentence = sentence.strip()
	return sentence

# Converts given sentence to matrix.
def sentence2matrix(hParams, x, vocab, model_embedding, is_input):
	x = apply_filter(x.split(" "))

	if hParams.reverse_input_sequence and is_input:
		x = x[::-1]

	r = []
	for xx in x:
		if hParams.embedding_type == "fasttext":
			try:
				r.append(model_embedding.wv[xx])
			except:
				r.append(model_embedding.wv[hParams.vocab_special_token[3]])
		else:
			try:
				r.append(vocab.index(xx))
			except:
				r.append(vocab.index(hParams.vocab_special_token[3]))

	r = np.array(r)
	r = r.reshape((1,) + r.shape)
	return r

# Masks unwanted sequence in array.
def sequence_mask(matrix, mask_index):
	assert len(matrix.shape) == 2

	seq_mask = []
	for x in range(0, matrix.shape[0]):
		seq_mask_timestep = []
		valid = True

		for t in range(0, matrix.shape[1]):
			if matrix[x][t] == mask_index and valid == True:
				valid = False
			seq_mask_timestep.append(valid)

		seq_mask.append(seq_mask_timestep)
	seq_mask = np.array(seq_mask)
	return seq_mask

# Seperates given words from Embedding model and creates seperate KeyedVectors object.
def embedding_seperate(path, words, model_embedding, hParams):
	model_embedding_kv = KeyedVectors(hParams.embedding_size)

	for word in words:
		try:
			model_embedding_kv.add([word], [model_embedding.wv[word]])
		except:
			pass

	model_embedding_kv.save(path)
	return model_embedding_kv

# Trains Embedding with given data.
def embedding_train(path, hParams, all_data):
	all_data = (" ".join(hParams.vocab_special_token) + "\n" + all_data.replace("\t", "\n"))

	all_sentences = apply_filter([apply_filter(cumle.split(" ")) for cumle in apply_filter(all_data.split("\n"))])

	if hParams.embedding_type == "fasttext":
		print("[*] Training FastText..")
		model_embedding = FastText(
			size=hParams.embedding_size,
			window=hParams.ngram,
			min_count=0,
			workers=3,
			sorted_vocab=1
		)
		model_embedding.build_vocab(all_sentences)
		model_embedding.train(all_sentences, total_examples=model_embedding.corpus_count, epochs=1)

	elif hParams.embedding_type == "word2vec":
		print("[*] Training Word2Vec..")

		vocab_limit = hParams.vocab_limit
		if vocab_limit == 0:
			vocab_limit = None

		model_embedding = Word2Vec(
			sentences=all_sentences,
			size=hParams.embedding_size,
			window=hParams.ngram,
			min_count=0,
			workers=3,
			sorted_vocab=1,
			max_final_vocab=vocab_limit,
			compute_loss=True
		)
		print("[*] Word2Vec initial loss", model_embedding.get_latest_training_loss())
		model_embedding.train(all_sentences, total_examples=model_embedding.corpus_count, epochs=1)
		print("[*] Word2Vec final loss", model_embedding.get_latest_training_loss())
	
	model_embedding.save(path + "_" + str(hParams.embedding_type))
	print("[*] Embedding model saved to {}".format(path + "_" + str(hParams.embedding_type)))
	return (model_embedding, model_embedding.wv.syn0, model_embedding.wv)

# Loads pre-trained FastText model. (.bin and .vec format).
def _embedding_load_pre_fasttext(pretrained_path):
	try:
		model_embedding = FastText.load_fasttext_format(pretrained_path)
		print("[+] FastText Embedding model successfully loaded from {}".format(pretrained_path))
		return model_embedding
	except:
		raise FileNotFoundError("[!] FastText Embedding model couldn't be loaded from {}".format(pretrained_path))

# Loads pre-trained Word2Vec model.
def _embedding_load_pre_word2vec(pretrained_path):
	try:
		model_embedding = Word2Vec.load(pretrained_path)
		print("[+] Word2Vec Embedding model successfully loaded from {}".format(pretrained_path))
		return model_embedding
	except:
		raise FileNotFoundError("[!] Word2Vec Embedding model couldn't be loaded from {}".format(pretrained_path))

# Loads pre-trained model.
def _embedding_load_trained(path, hParams):
	try:
		if hParams.embedding_type == "fasttext":
			model_embedding = FastText.load(path + "_" + hParams.embedding_type)
		elif hParams.embedding_type == "word2vec":
			model_embedding = Word2Vec.load(path + "_" + hParams.embedding_type)

		print("[+] Embedding model successfully loaded from {}".format(path + "_" + hParams.embedding_type))
		return model_embedding
	except:
		raise FileNotFoundError("[!] Embedding model couldn't be loaded from {}".format(path + "_" + hParams.embedding_type))

# Loads Embedding model from disk.
def embedding_load(path, hParams, all_data):
	if hParams.embedding_use_pretrained:
		if hParams.embedding_type == "fasttext":
			model_embedding = _embedding_load_pre_fasttext(hParams.embedding_pretrained_path)
		elif hParams.embedding_type == "word2vec":
			model_embedding = _embedding_load_pre_word2vec(hParams.embedding_pretrained_path)

		words = hParams.vocab_special_token + sorted(list(set([word for pair in apply_filter(all_data.split("\n")) for word in apply_filter(pair.replace("\t", " ").split(" "))])))
		model_embedding_kv = embedding_seperate("model/EmbeddingKeyedVector", words, model_embedding, hParams)
		embedding_matrix = model_embedding_kv.syn0
	else:
		model_embedding = _embedding_load_trained(path, hParams)
		model_embedding_kv = model_embedding.wv
		embedding_matrix = model_embedding_kv.syn0
	return (model_embedding, embedding_matrix, model_embedding_kv)

# Data generator class.
# For each call, takes a slice from given data matrices and returns them.
# When it comes to end, it resets back again.
class DataGenerator:
	def __init__(self, encX, decX, decy, batch_size, bucket_structure, data_count):
		self.encX = encX
		self.decX = decX
		self.decy = decy
		self.batch_size = batch_size
		self.bucket_structure = bucket_structure
		self.data_count = data_count

		self.data = 0
		self.bucket = 0
	def __call__(self):
		if self.bucket >= len(self.bucket_structure):
			self.bucket = 0
		while self.encX[self.bucket].shape[0] == 0 and self.decX[self.bucket].shape[0] == 0 and self.decy[self.bucket].shape[0] == 0:
			self.bucket += 1
			if self.bucket >= len(self.bucket_structure):
				self.bucket = 0

		if self.data_count[self.bucket] <= (self.data+self.batch_size) and self.data_count[self.bucket] >= self.data:
			enc_x = self.encX[self.bucket][self.data:]
			dec_x = self.decX[self.bucket][self.data:]
			dec_y = self.decy[self.bucket][self.data:]

			self.data = 0
			self.bucket += 1
		else:
			enc_x = self.encX[self.bucket][self.data:(self.data+self.batch_size)]
			dec_x = self.decX[self.bucket][self.data:(self.data+self.batch_size)]
			dec_y = self.decy[self.bucket][self.data:(self.data+self.batch_size)]
			self.data += self.batch_size
		return enc_x, dec_x, dec_y

# Load JSON config file and create HParams class.
# And check if parameters is invalid or not.
def load_parameters(file):
	hParams = tf.contrib.training.HParams(
		rnn_unit=[None],
		rnn_cell=None,
		encoder_rnn_type=None,
		attention_mechanism=None,
		attention_size=None,
		dense_layers=[None],
		dense_activation=None,
		optimizer=None,
		learning_rate=None,
		dropout_keep_prob_dense=None,
		dropout_keep_prob_rnn_input=None,
		dropout_keep_prob_rnn_output=None,
		dropout_keep_prob_rnn_state=None,
		bucket_use_padding=None,
		bucket_padding_input=[None],
		bucket_padding_output=[None],
		train_epochs=None,
		train_steps=None,
		train_batch_size=None,
		log_per_step_percent=None,
		embedding_use_pretrained=None,
		embedding_pretrained_path=None,
		embedding_type=None,
		embedding_size=None,
		embedding_negative_sample=None,
		vocab_limit=None,
		vocab_special_token=[None],
		ngram=None,
		reverse_input_sequence=None,
		seq2seq_loss=None
	).parse_json(open(file, "r").read())

	assert isinstance(hParams.rnn_unit, list)
	assert hParams.rnn_cell in ["lstm", "gru"]
	assert hParams.encoder_rnn_type in ["unidirectional", "bidirectional"]
	assert hParams.attention_mechanism in ["bahdanau", "luong", None]
	assert isinstance(hParams.dense_layers, list)
	assert hParams.dense_activation in ["relu", "sigmoid", "tanh", None]
	assert hParams.optimizer in ["sgd", "adam", "rmsprop"]
	assert hParams.dropout_keep_prob_dense > 0.0 and hParams.dropout_keep_prob_dense <= 1.0
	assert hParams.dropout_keep_prob_rnn_input > 0.0 and hParams.dropout_keep_prob_rnn_input <= 1.0
	assert hParams.dropout_keep_prob_rnn_output > 0.0 and hParams.dropout_keep_prob_rnn_output <= 1.0
	assert hParams.dropout_keep_prob_rnn_state > 0.0 and hParams.dropout_keep_prob_rnn_state <= 1.0
	assert hParams.embedding_type in ["fasttext", "word2vec"]
	assert len(hParams.vocab_special_token) == 4

	if hParams.encoder_rnn_type == "bidirectional" and hParams.attention_mechanism == None:
		raise Exception("Encoder Bi-RNN cannot be used without attention mechanism.")

	return hParams

# Prepares bucket structure and data count of each bucket.
def prepare_parameters(hParams, all_data):
	all_data_pair = apply_filter(all_data.split("\n"))
	all_data_pair = [apply_filter(pair.split("\t")) for pair in all_data_pair if len(apply_filter(pair.split("\t"))) == 2]

	bucket_structure = []
	if hParams.bucket_use_padding:
		bucket_input = sorted(hParams.bucket_padding_input)
		bucket_output = sorted(hParams.bucket_padding_output)
	else:
		bucket_input = []
		bucket_output = []

		for (_input, _output) in all_data_pair:
			_input = apply_filter(_input.split(" "))
			_output = apply_filter((hParams.vocab_special_token[0] + " " + _output + " " + hParams.vocab_special_token[1]).split(" "))

			bucket_input.append(len(_input))
			bucket_output.append(len(_output))

		bucket_input = sorted(list(set(bucket_input)))
		bucket_output = sorted(list(set(bucket_output)))

		if len(bucket_input) != len(bucket_output):
			if bucket_input > bucket_output:
				bucket_input = bucket_input[:len(bucket_output)]
			if bucket_output > bucket_input:
				bucket_output = bucket_output[:len(bucket_input)]

	for bg in bucket_input:
			for bc in bucket_output:
				bucket_structure.append((bg, bc))

	data_count = []
	if hParams.bucket_use_padding:
		for (bg, bc) in bucket_structure:
			data_count_cur_bucket = 0
			for (_input, _output) in all_data_pair:
				_input = apply_filter(_input.split(" "))
				_output = apply_filter((hParams.vocab_special_token[0] + " " + _output + " " + hParams.vocab_special_token[1]).split(" "))

				bucketIndex = -1
				for b in range(0, len(bucket_structure)):
					bucket = bucket_structure[b]
					
					if len(_input) <= bucket[0] and len(_output) <= bucket[1]:
						bucketIndex = b
						break

				if bucketIndex == -1:
					continue

				input_gap = bucket_structure[bucketIndex][0] - len(_input)
				output_gap = bucket_structure[bucketIndex][1] - len(_output)

				if len(_input)+input_gap == bg and len(_output)+output_gap == bc:
					data_count_cur_bucket += 1
			data_count.append(data_count_cur_bucket)
	else:
		for (bg, bc) in bucket_structure:
			data_count_cur_bucket = 0
			for (_input, _output) in all_data_pair:
				_input = apply_filter(_input.split(" "))
				_output = apply_filter((hParams.vocab_special_token[0] + " " + _output + " " + hParams.vocab_special_token[1]).split(" "))

				if len(_input) == bg and len(_output) == bc:
					data_count_cur_bucket += 1
			data_count.append(data_count_cur_bucket)

	return bucket_structure, data_count