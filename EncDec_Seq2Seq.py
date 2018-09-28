#! -*- coding: UTF-8 -*-
"""
Main class for Sequence-to-Sequence Encoder-Decoder model.
"""

import tensorflow as tf

class EncDec_Seq2Seq:
	def __init__(self, mode, hParams, embedding_matrix, VOCAB, vocab_length, decode_max_timestep, sess_cfg=None):
		self.mode = mode
		assert self.mode in ["train", "test"]
		
		self.hParams = hParams
		self.embedding_matrix = embedding_matrix
		self.VOCAB = VOCAB
		self.vocab_length = vocab_length

		# Set JSON config file's attributes as own attribute.
		for key, value in zip(list(self.hParams.values().keys()), list(self.hParams.values().values())):
			self.__setattr__(key, value)

		if self.mode == "test":
			self.decode_max_timestep = decode_max_timestep
		else:
			self.decode_max_timestep = None

		# Add last layer to dense for output of vocab.
		self.dense_layers.append(self.vocab_length)

		if self.dense_activation == "relu":
			self.dense_activation = tf.nn.relu
		elif self.dense_activation == "sigmoid":
			self.dense_activation = tf.nn.sigmoid
		elif self.dense_activation == "tanh":
			self.dense_activation = tf.nn.tanh
		if self.optimizer == "sgd":
			self.optimizer = tf.train.GradientDescentOptimizer
		elif self.optimizer == "adam":
			self.optimizer = tf.train.AdamOptimizer
		elif self.optimizer == "rmsprop":
			self.optimizer = tf.train.RMSPropOptimizer

		# If attention size is not specified, specify it as unit of RNN's last layer.
		if self.attention_size == None:
			self.attention_size = self.rnn_unit[-1]

		if sess_cfg != None:
			self.sess = tf.Session(config=sess_cfg)
		else:
			self.sess = tf.Session()

	# Build Input tensors.
	def build_placeholder(self):
		self.embedding_matrix = tf.constant(self.embedding_matrix, dtype=tf.float32)

		if self.embedding_type == "fasttext":
			self.enc_xx_n = tf.placeholder(tf.float32, shape=(None, None, self.embedding_matrix.shape[1]))
			self.enc_xx = self.enc_xx_n
		elif self.embedding_type == "word2vec":
			self.enc_xx_n = tf.placeholder(tf.int32, shape=(None, None))
			self.enc_xx = tf.nn.embedding_lookup(self.embedding_matrix, self.enc_xx_n)
			self.enc_xx_o = tf.one_hot(self.enc_xx_n, self.vocab_length)

		self.dec_xx_n = tf.placeholder(tf.int32, shape=(None, None))
		self.dec_yy_n = tf.placeholder(tf.int32, shape=(None, None))

		self.dec_xx = tf.nn.embedding_lookup(self.embedding_matrix, self.dec_xx_n)
		self.dec_yy_o = tf.one_hot(self.dec_yy_n, self.vocab_length)

		if self.seq2seq_loss:
			self.seq_loss_weight = tf.placeholder(tf.float32, shape=(None, None))

	# Build Encoder component.
	def build_encoder(self):
		with tf.name_scope("encoder"):
			if self.rnn_cell == "lstm":
				self.rnn_cell_fn = tf.nn.rnn_cell.LSTMCell
			elif self.rnn_cell == "gru":
				self.rnn_cell_fn = tf.nn.rnn_cell.GRUCell

			if self.encoder_rnn_type == "unidirectional":
				self.enc_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell_fn(unit) for unit in self.rnn_unit])

				# Dropout applied in training.
				if self.mode == "train":
					self.enc_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
						self.enc_rnn_cell,
						input_keep_prob=self.dropout_keep_prob_rnn_input,
						output_keep_prob=self.dropout_keep_prob_rnn_output,
						state_keep_prob=self.dropout_keep_prob_rnn_state
					)

				self.e_out, self.e_state = tf.nn.dynamic_rnn(self.enc_rnn_cell, self.enc_xx, dtype=tf.float32)
				if self.attention_mechanism != None:
					self.attention_memory = self.e_out

			elif self.encoder_rnn_type == "bidirectional":
				self.enc_rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell_fn(unit) for unit in self.rnn_unit])
				self.enc_rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell_fn(unit) for unit in self.rnn_unit])

				# Dropout applied in training.
				if self.mode == "train":
					self.enc_rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
						self.enc_rnn_cell_fw,
						input_keep_prob=self.dropout_keep_prob_rnn_input,
						output_keep_prob=self.dropout_keep_prob_rnn_output,
						state_keep_prob=self.dropout_keep_prob_rnn_state
					)
					self.enc_rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
						self.enc_rnn_cell_bw,
						input_keep_prob=self.dropout_keep_prob_rnn_input,
						output_keep_prob=self.dropout_keep_prob_rnn_output,
						state_keep_prob=self.dropout_keep_prob_rnn_state
					)

				self.e_out, self.e_state = tf.nn.bidirectional_dynamic_rnn(self.enc_rnn_cell_fw, self.enc_rnn_cell_bw, self.enc_xx, dtype=tf.float32)
				if self.attention_mechanism != None:
					self.attention_memory = tf.concat(self.e_out, axis=2)

	# Build Decoder component.
	def build_decoder(self):
		with tf.name_scope("decoder"):
			if self.rnn_cell == "lstm":
				self.rnn_cell_fn = tf.nn.rnn_cell.LSTMCell
			elif self.rnn_cell == "gru":
				self.rnn_cell_fn = tf.nn.rnn_cell.GRUCell

			self.dec_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell_fn(unit) for unit in self.rnn_unit])

			# Dropout applied in training.
			if self.mode == "train":
				self.dec_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
					self.dec_rnn_cell, 
					input_keep_prob=self.dropout_keep_prob_rnn_input,
					output_keep_prob=self.dropout_keep_prob_rnn_output,
					state_keep_prob=self.dropout_keep_prob_rnn_state
				)

			if self.attention_mechanism != None:
				if self.attention_mechanism == "bahdanau":
					self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.attention_size, self.attention_memory)
				elif self.attention_mechanism == "luong":
					self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.attention_size, self.attention_memory)
				
				self.dec_final_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_rnn_cell, self.attention_mechanism)
			else:
				self.dec_final_cell = self.dec_rnn_cell

			self.dec_initial_state = self.dec_final_cell.zero_state(tf.shape(self.enc_xx)[0], dtype=tf.float32)
			if self.encoder_rnn_type == "unidirectional":
				if self.attention_mechanism != None:
					self.dec_initial_state = self.dec_initial_state.clone(cell_state=self.e_state)
				else:
					self.dec_initial_state = self.e_state

			# In training, next timestep input of Decoder is already specified.
			if self.mode == "train":
				self.helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.dec_xx, sequence_length=[tf.shape(self.dec_xx)[1]])

			# In testing, next timestep input of Decoder is current timestep's output word.
			if self.mode == "test":
				self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_matrix, start_tokens=tf.ones([tf.shape(self.enc_xx)[0]], dtype=tf.int32)*self.VOCAB.index(self.vocab_special_token[0]), end_token=self.VOCAB.index(self.vocab_special_token[1]))

			self.decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.dec_final_cell, helper=self.helper, initial_state=self.dec_initial_state)
			self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=self.decoder, maximum_iterations=self.decode_max_timestep)
			self.lastOutput = self.outputs.rnn_output

			# Build Dense layer at Decoder's each timestep.
			for dl in range(0, len(self.dense_layers)-1):
				self.lastOutput = tf.layers.dense(self.lastOutput, units=self.dense_layers[dl], activation=self.dense_activation)
				if self.mode == "train":
					self.lastOutput = tf.layers.dropout(self.lastOutput, rate=(1.0-self.dropout_keep_prob_dense))

			self.lastOutput = tf.layers.dense(self.lastOutput, units=self.dense_layers[-1], activation=None)
			self.prediction = tf.nn.softmax(self.lastOutput)

	# Initialize training tensors.
	def train_op(self):
		# Initialize loss.
		if self.seq2seq_loss:
			self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.lastOutput, targets=self.dec_yy_n, weights=self.seq_loss_weight)
		else:
			self.loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=self.lastOutput, onehot_labels=self.dec_yy_o))

		self.train = self.optimizer(self.learning_rate).minimize(self.loss)

	# Trains the model for given batch.
	def train_batch(self, batch_enc_xx_n, batch_dec_xx_n, batch_dec_yy_n, s2s_loss_weight=None):
		if self.mode == "train":
			feed = {
				self.enc_xx_n:batch_enc_xx_n,
				self.dec_xx_n:batch_dec_xx_n,
				self.dec_yy_n:batch_dec_yy_n
			}
			if not np.array_equal(s2s_loss_weight, None) and self.seq2seq_loss:
				feed.update({self.seq_loss_weight:s2s_loss_weight})
			return self.sess.run(self.train, feed_dict=feed)
		else:
			raise ValueError("[!] Model cannot be trained without train mode.")

	# Predicts given data.
	def predict(self, batch_enc_xx_n, batch_dec_xx_n, batch_dec_yy_n):
		if self.mode == "train":
			feed = {
				self.enc_xx_n:batch_enc_xx_n,
				self.dec_xx_n:batch_dec_xx_n,
				self.dec_yy_n:batch_dec_yy_n
			}

			if self.embedding_type == "fasttext":
				return self.sess.run([self.dec_yy_o, self.prediction], feed_dict=feed)
			elif self.embedding_type == "word2vec":
				return self.sess.run([self.enc_xx_o, self.dec_yy_o, self.prediction], feed_dict=feed)

	def predict_infer(self, batch_enc_xx_n):
		feed = {
			self.enc_xx_n:batch_enc_xx_n
		}
		return self.sess.run(self.prediction, feed_dict=feed)

	# Returns given data's encoded state.
	def predict_state(self, batch_enc_xx_n):
		feed = {
			self.enc_xx_n:batch_enc_xx_n
		}
		return self.sess.run([self.e_state], feed_dict=feed)

	# Calculates loss..
	def calculate_loss(self, batch_enc_xx_n, batch_dec_xx_n, batch_dec_yy_n, s2s_loss_weight=None):
		if self.mode == "train":
			feed = {
				self.enc_xx_n:batch_enc_xx_n,
				self.dec_xx_n:batch_dec_xx_n,
				self.dec_yy_n:batch_dec_yy_n
			}
			if not np.array_equal(s2s_loss_weight, None) and self.seq2seq_loss:
				feed.update({self.seq_loss_weight:s2s_loss_weight})
			return self.sess.run(self.loss, feed_dict=feed)

	# Load weights from disk.
	def load(self, path, force):
		try:
			self.saver.restore(self.sess, path)
			print("[+] Weights loaded successfully!")
		except Exception as e:
			if force:
				raise e
			
			print("[***] Weights couldn't be loaded from {}".format(path))
			if self.mode == "train":
				print("[*] Model will be trained from scratch.")

	# Save weights to disk.
	def save(self, path):
		self.saver.save(self.sess, path)
		print("[+] Weights saved.")

	# Build model components and initialize weights.
	def build(self):
		self.build_placeholder()
		self.build_encoder()
		self.build_decoder()

		if self.mode == "train":
			self.train_op()

		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

		print("[+] Model build done.")