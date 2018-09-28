#! -*- coding: UTF-8 -*-
"""
Build model, if exists load weights. Then train the model on dataset.
"""

import numpy as np
import random
from EncDec_Seq2Seq import EncDec_Seq2Seq
from utils import *

# Prepare parameters and embedding.
hParams = load_parameters("model.json")
all_data = open("data/all_data.txt", "r", encoding="utf-8").read()
bucket_structure, data_count = prepare_parameters(hParams, all_data)
model_embedding, embedding_matrix, keyed_vector = embedding_load("model/EmbeddingModel", hParams, all_data)

VOCAB = list(keyed_vector.vocab.keys())
vocab_length = len(VOCAB)

encX = np.load("data/encX.npy")
decX = np.load("data/decX.npy")
decy = np.load("data/decy.npy")

# Build model class.
model = EncDec_Seq2Seq(
	mode="train",
	hParams=hParams,
	embedding_matrix=embedding_matrix,
	VOCAB=VOCAB,
	vocab_length=vocab_length,
	decode_max_timestep=bucket_structure[-1][1]
)

# Build model components and initialize all weights.
model.build()

# If trained weights exist on disk, load it.
model.load("model/Weights", False)

# Build data generator class.
data_generate = DataGenerator(encX, decX, decy, hParams.train_batch_size, bucket_structure, data_count)

# Train!
for epoch in range(0, hParams.train_epochs):
	for step in range(0, hParams.train_steps):
		# Take a mini-batch from data.
		enc_x, dec_x, dec_y = data_generate()

		# Ignore unwanted words (like <pad>) for loss calculation.
		seq_mask = sequence_mask(dec_y, VOCAB.index(hParams.vocab_special_token[2]))

		# Train current mini-batch.
		model.train_batch(enc_x, dec_x, dec_y, seq_mask)

		# Logging for progress.
		if step%(hParams.train_steps/hParams.log_per_step_percent) == 0:
			print("Epoch {} | Step {} | Batch Loss {}".format(
					epoch,
					step,
					model.calculate_loss(enc_x, dec_x, dec_y, seq_mask)
				)
			)

			# Select random data from batch and print that out.
			ri = random.randint(0, enc_x.shape[0]-1)
			enc_x, dec_x, dec_y = enc_x[ri], dec_x[ri], dec_y[ri]
			enc_x = enc_x.reshape((1,) + enc_x.shape)
			dec_x = dec_x.reshape((1,) + dec_x.shape)
			dec_y = dec_y.reshape((1,) + dec_y.shape)

			_target, _output = model.predict(enc_x, dec_x, dec_y)

			if hParams.embedding_type == "fasttext":
				_inp = decode_seq_vector(enc_x[0], model_embedding, hParams)
			elif hParams.embedding_type == "word2vec":
				_inp = decode_seq(enc_x[0], VOCAB)
			_tar = decode_seq(_target[0], VOCAB)
			_out = decode_seq(_output[0], VOCAB)

			print("Input: {}".format(_inp))
			print("Target: {}".format(_tar))
			print("Output: {}".format(_out))
			print("---------------------------------------")

	# Save model every epoch.
	print("[Checkpoint] Epoch {}".format(str(epoch)))
	model.save("model/Weights")