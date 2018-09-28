#! -*- coding: UTF-8 -*-

import numpy as np
import json, operator
from utils import *

# Creates 3 matrices with given config & data: encX, decX, decy
def create_matrix(all_data, vocab, bucket_structure, hParams, model_embedding):
	encX = []
	decX = []
	decy = []

	inout_pairs = apply_filter(all_data.split("\n"))

	for b in range(0, len(bucket_structure)):
		encX.append([])
		decX.append([])
		decy.append([])

	for pair in inout_pairs:
		enc_xx = []
		dec_xx = []
		dec_yy = []
		
		pair = pair.split("\t")
		_input = apply_filter(pair[0].split(" "))
		_output = apply_filter((hParams.vocab_special_token[0] + " " + pair[1] + " " + hParams.vocab_special_token[1]).split(" "))

		if len(_output) <= 2:
			continue

		if hParams.reverse_input_sequence:
			_input = _input[::-1]

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

		for i in range(0, input_gap):
			if hParams.embedding_type == "word2vec":
				enc_xx.append(vocab.index(hParams.vocab_special_token[2]))
			elif hParams.embedding_type == "fasttext":
				enc_xx.append(model_embedding.wv[hParams.vocab_special_token[2]])
		for i in range(0, len(_input)):
			if hParams.embedding_type == "word2vec":
				try:
					enc_xx.append(vocab.index(_input[i]))
				except:
					enc_xx.append(vocab.index(hParams.vocab_special_token[3]))
			elif hParams.embedding_type == "fasttext":
				try:
					enc_xx.append(model_embedding.wv[_input[i]])
				except:
					enc_xx.append(model_embedding.wv[hParams.vocab_special_token[3]])

		for i in range(0, len(_output)):
			if i < len(_output)-1:
				try:
					dec_xx.append(vocab.index(_output[i]))
				except:
					dec_xx.append(vocab.index(hParams.vocab_special_token[3]))
				try:
					dec_yy.append(vocab.index(_output[i+1]))
				except:
					dec_yy.append(vocab.index(hParams.vocab_special_token[3]))
		for i in range(0, output_gap):
			dec_xx.append(vocab.index(hParams.vocab_special_token[2]))
			dec_yy.append(vocab.index(hParams.vocab_special_token[2]))

		if len(enc_xx) > 0 and len(dec_xx) > 0 and len(dec_yy) > 0:
			enc_xx = np.array(enc_xx)
			dec_xx = np.array(dec_xx)
			dec_yy = np.array(dec_yy)

			encX[bucketIndex].append(enc_xx)
			decX[bucketIndex].append(dec_xx)
			decy[bucketIndex].append(dec_yy)

	for x in range(0, len(encX)):
		encX[x] = np.array(encX[x])
		decX[x] = np.array(decX[x])
		decy[x] = np.array(decy[x])
	encX = np.array(encX)
	decX = np.array(decX)
	decy = np.array(decy)

	print("[*] Data matrix successfully created.")
	return encX, decX, decy

# Prepare parameters.
hParams = load_parameters("model.json")
all_data = open("data/all_data.txt", "r", encoding="utf-8").read()
bucket_structure, data_count = prepare_parameters(hParams, all_data)

# Load Embedding model.
if hParams.embedding_use_pretrained:
	model_embedding, embedding_matrix, keyed_vector = embedding_load("model/EmbeddingModel", hParams, all_data)
else:
	model_embedding, embedding_matrix, keyed_vector = embedding_train("model/EmbeddingModel", hParams, all_data)

VOCAB = list(keyed_vector.vocab.keys())
vocab_length = len(VOCAB)

# Create data matrices.
encX, decX, decy = create_matrix(all_data, VOCAB, bucket_structure, hParams, model_embedding)

# Save to disk.
vocabF = open("data/vocab.txt", "w", encoding="utf-8")
for word in VOCAB:
	vocabF.write(word + "\n")
vocabF.close()

np.save("data/encX", encX)
np.save("data/decX", decX)
np.save("data/decy", decy)

print("Example count:", len(apply_filter(all_data.split("\n"))))
print("Vocabulary length:", vocab_length)
print("Matrix sizes:")
print("encX", encX.shape)
print("decX", decX.shape)
print("decy", decy.shape)