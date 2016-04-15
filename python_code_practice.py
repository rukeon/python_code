#1 make hash by one code

keys = ['a', 'b', 'c']
values = [1,2,3]

hash = {k:v for k,v in zip(keys, values)}

#2 make reducer
def myreducer(fnc, seq):
	tally = seq[0]
	for n in seq[1:]:
		tally = fnc(tally, n)
	return tally

myreducer( (lambda x, y: x*y), [1,2,3,4])

#3 squared sum
def ss(nums):
	return sum(x**2 for x in nums)
