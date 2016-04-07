import random

total_list = []
for j in range(100):
	count = 0
	num_set = set()
	for i in range(10):
		pred = random.randrange(1, 9, 1)
		if pred in num_set:
			count += 1
		num_set.add(pred)
	total_list.append(count/10)

# for test
sum(total_list)/len(total_list)


