# First generate the integer then square with user defined func

def integers(f):
	i = 1
	while True:
		yield i
		i = f(i)

def squares(f):
	for i in integers(f):
		yield i*i

def take(n, seq):
	seq = iter(seq)
	result = []
	try:
		for i in range(n):
			result.append(next(seq))
	except StopIteration:
		pass
	return result

def f(i):
	return 2*i

print(take(5, squares(f)))

# it returns [1, 4, 16, 64, 256]


# Second shrink common code by generater

# common code
def cat(filenames):
	for f in filenames:
		for line in open(f):
			print line

def grep(pattern, filenames):
	for f in filenames:
		for line in open(f):
			if pattern in line:
				print line

# Both these programs have lot of code in common. But with generators makes it possible to do it.

def readfiles(filenames):
	for f in filenames:
		for line in open(f):
			yield line

def grep(pattern, lines):
	return (line for line in lines if pattern in line)

def printlines(lines):
	for line in lines:
		print line

def main(pattern, filenames):
	lines = readfiles(filenames)
	lines = grep(pattern, lines)
	printlines(lines)
