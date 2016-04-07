import random

#  random.sample(range(1, 10), 3)

rn = ["0", "0", "0"]
rn[0] = str(random.randrange(1, 9, 1))
rn[1] = rn[0]
rn[2] = rn[0]
while (rn[0] == rn[1]):
	rn[1] = str(random.randrange(1, 9, 1))
while (rn[0] == rn[2] or rn[1] == rn[2]):
	rn[2] = str(random.randrange(1, 9, 1))

t_cnt = 0
s_cnt = 0
b_cnt = 0

print("game start!")
print("-----------------------")

while( s_cnt < 3):
	num = str(input("give me a number: "))
	
	s_cnt = 0
	b_cnt = 0

	for i in range(0, 3):
		for j in range(0, 3):
			if(num[i] == str(rn[j]) and i == j):
				s_cnt += 1
			elif(num[i] == str(rn[j]) and i != j):
				b_cnt += 1
	print("result : [", s_cnt, "] Strike [" , b_cnt, "] Ball")
	t_cnt += 1

print("-------------------------------------")
print(t_cnt, " times, you tri")
