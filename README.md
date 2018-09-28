# tf_encdec_seq2seq
Configurable advanced Encoder-Decoder Sequence-to-Sequence model. Built with TensorFlow.

# Features
* Easily configurable.
* Unidirectional-RNN, Bidirectional-RNN
* Attention models.
* Bucketing.
* Embedding models.

# Requirements
+ <a href="https://github.com/tensorflow/tensorflow">tensorflow</a>
+ <a href="https://github.com/numpy/numpy">numpy</a>
+ <a href="https://github.com/RaRe-Technologies/gensim/">gensim</a>

`pip install -r requirements.txt`

# Preparing Data
* Put your TSV file under <b>data/</b> directory as <b>all_data.txt</b> Each line in file is input-output pair, seperated with tab.
<br>Then run `python build_data_matrix.py`
<br>It will create your data matrices through your raw data. If pre-trained is not used, it will train an Embedding model.

# Train
*cough* `python train.py` *cough*

# Interactive mode & Inference through file
`python interactive.py`<br>
`python test.py my_input_sentences.txt`

# Config
* <b>rnn_unit</b> | List | Specifies unit count of each layer on Encoder and Decoder.
* <b>rnn_cell</b> | String | RNN cell type of Encoder and Decoder.<br>[LSTM, GRU]
* <b>encoder_rnn_type</b> | String | Encoder's RNN type.<br>[unidirectional, bidirectional]
* <b>attention_mechanism</b> | String | Attention mechanism of the model.<br>[luong, bahdanau, None]
* <b>attention_size</b> | int | Attention size of the model. (If not specified, will be defined as rnn_unit's last element)
* <b>dense_layers</b> | List | Specifies unit count of each layer on FC.
* <b>dense_activation</b> | String | Activation function to be used on FC layer.<br>[relu, sigmoid, tanh, None]
* <b>optimizer</b> | String | Optimizer function.<br>[sgd, adam, rmsprop]
* <b>learning_rate</b> | Float | Learning rate.
* <b>dropout_keep_prob_dense</b> | Float | Dropout keep-prob rate on FC layer. (> 0.0, <= 1.0)
* <b>dropout_keep_prob_rnn_input</b> | Float | Dropout keep-prob rate on RNN input. (> 0.0, <= 1.0)
* <b>dropout_keep_prob_rnn_output</b> | Float | Dropout keep-prob rate on RNN output. (> 0.0, <= 1.0)
* <b>dropout_keep_prob_rnn_state</b> | Float | Dropout keep-prob rate on RNN state. (> 0.0, <= 1.0)
* <b>bucket_use_padding</b> | Bool | If true, adds <b>\<pad\></b> tags to input and output sentence. So reduces count of buckets.
* <b>bucket_padding_input</b> | List | Bucket sizes of input.
* <b>bucket_padding_output</b> | List | Bucket sizes of output.
* <b>train_epochs</b> | int | Epochs to be passed during training model. <b>(Each epoch saves model to disk.)</b>
* <b>train_steps</b> | int | Steps to be passed during training model.
* <b>train_batch_size</b> | int | Batch-size during training.
* <b>log_per_step_percent</b> | int | Percent value that will be used as progress log point.
* <b>embedding_use_pretrained</b> | Bool | Use pre-trained Embedding or not.
* <b>embedding_pretrained_path</b> | String | Path of the pre-trained Embedding files.
* <b>embedding_type</b> | String | Embedding type of the model.<br>[word2vec, fasttext]
* <b>embedding_size</b> | int | Embedding size of the model.
* <b>embedding_negative_sample</b> | int | Embedding negative sampling value.
* <b>vocab_limit</b> | int | Vocabulary limit during build Embedding model.
* <b>vocab_special_token</b> | List | Special vocabulary tokens that will be used as padding tag, unknown words, start and end of the sentences.
* <b>ngram</b> | int | N-gram value of the Embedding model.
* <b>reverse_input_sequence</b> | Bool | If true, reverse words of the input sentence.
* <b>seq2seq_loss</b> | Bool | Use seq2seq loss. That means, during loss calculation tags like <b>\<pad\></b> going to be ignored.
