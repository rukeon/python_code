# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/rnn/ptb/reader.py

import collections

def _read_words(filename):
	with gfile.GFile(filename, "r") as f:
		return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id

