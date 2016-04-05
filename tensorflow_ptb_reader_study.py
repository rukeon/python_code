# explanation of _build_vocab
import collections

def _read_words(filename):
	with gfile.Gfile(filename, "r) as f:
		return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
	data = _read_words(filename)
	
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x : (-x[1]. x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	
	return word_to_id

# explanation)
# sample = "hello my name is ki hyun. ki hyun hello my my my mym\n"
# 
# sample_list = sample.replace("\n", " <eos>").split()
# 
# counter = collections.Counter(sample_list)
# -> Counter({'my': 4, 'ki': 2, 'hello': 2, 'mym': 1, 'name': 1, 'is': 1, '<eos>': 1, 'hyun': 1, 'hyun.': 1})
#
# count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
# -> [('my', 4),
#  ('hello', 2),
#  ('ki', 2),
#  ('<eos>', 1),
#  ('hyun', 1),
#  ('hyun.', 1),
#  ('is', 1),
#  ('mym', 1),
#  ('name', 1)]
# words, _= list(zip(*count_pairs))
# -> words = ('my', 'hello', 'ki', '<eos>', 'hyun', 'hyun.', 'is', 'mym', 'name')
# - -> (4, 2, 2, 1, 1, 1, 1, 1, 1)
# word_to_id = dict(zip(words, range(len(words))))
# -> {'<eos>': 3,
#  'hello': 1,
#  'hyun': 4,
#  'hyun.': 5,
#  'is': 6,
#  'ki': 2,
#  'my': 0,
#  'mym': 7,
#  'name': 8}

