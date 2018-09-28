#! -*- coding: UTF-8 -*-
"""
Build model, load weights and let the user interact with model.
"""

import tensorflow as tf
from EncDec_Seq2Seq import EncDec_Seq2Seq
from utils import *

# Prepare parameters and embedding.
hParams = load_parameters("model.json")
all_data = open("data/all_data.txt", "r", encoding="utf-8").read()
bucket_structure, data_count = prepare_parameters(hParams, all_data)
model_embedding, embedding_matrix, keyed_vector = embedding_load("model/EmbeddingModel", hParams, all_data)

VOCAB = list(keyed_vector.vocab.keys())
vocab_length = len(VOCAB)

# Build model class.
model = EncDec_Seq2Seq(
	mode="test",
	hParams=hParams,
	embedding_matrix=embedding_matrix,
	VOCAB=VOCAB,
	vocab_length=vocab_length,
	decode_max_timestep=bucket_structure[-1][1],
	sess_cfg=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
)

# Build model components and initialize all weights.
model.build()

# Load weights.
model.load("model/Weights", True)

# Infer!
while True:
	inp = input("> ")
	if inp == "quit" or inp == "exit":
		break
	
	inp = clean_data(inp).strip()
	inp = pad_to_bucket(inp, bucket_structure, True, hParams)

	enc_x = sentence2matrix(hParams, inp, VOCAB, model_embedding, True)
	
	_output = model.predict_infer(enc_x)
	_output = decode_seq(_output[0], VOCAB)

	try:
		_output = _output[:_output.index(hParams.vocab_special_token[1])].strip()
	except:
		pass

	print(_output)