import numpy as np
import matplotlib.pyplot as plt
import csv


base_folder = '/home/aadi-z640/research/Imagined_counterfactuals/plots/'


x, y = [], []
with open(base_folder + 'reward_pop10_roll10_envrover_heterogeneous_4_2_seed2019-rewardglobal.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))

plt.plot(x, y, 'b')


x, y = [], []
with open(base_folder + 'reward_pop10_roll10_envrover_heterogeneous_6_3_seed2019-rewardglobal.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))



#fig, ax1 = plt.subplots()

plt.plot(x, y, 'g')


x, y = [], []

with open(base_folder + 'reward_pop10_roll10_envrover_heterogeneous_8_4_seed2019-rewardglobal.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))


plt.plot(x, y, 'orange')


'''
x, y = [], []

with open(base_folder + 'random_seed_2021/Result_2_million/score_Humanoid-v2_p10_s2021.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))


plt.plot(x, y, 'c')



x, y = [], []

with open(base_folder + 'random_seed_2022/Result_2_million/score_Humanoid-v2_p10_s2022.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))


plt.plot(x, y, 'r')
'''
'''
x, y = [], []
with open(base_folder + 'Humanoid_5_start_random/Results_random_seed_2022/Result_2_million/score_Humanoid-v2_p10_s2022.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))



#fig, ax1 = plt.subplots()

plt.plot(x, y, 'r')

'''

#fig, ax1 = plt.subplots()

'''
plt.rcParams.update({'font.size': 20})

plt.tick_params(labelsize=20)


color = 'tab:blue'

ax1.set_xlabel('Time Steps', fontsize=30)
ax1.set_ylabel('Average Reward (with exploration noise)', color=color, fontsize=20)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
'''


'''
color = 'tab:red'
ax2.set_ylabel('Average Reward (without exploration noise)', color=color, fontsize=20)  # we already handled the x-label with ax1
ax2.plot(x, y, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
'''


plt.rcParams.update({'font.size': 20})

plt.tick_params(labelsize=20)

plt.xlabel('Time Steps', fontsize=20)
plt.ylabel('Performance (Average Reward)', fontsize=20)
plt.title('MERL with heterogeneous agents (2 different types of agents)', fontsize=20)
plt.legend(['Coupling of 2 (4 rovers (2 of each type), 4POis)', 'Coupling of 3 (6 rovers (3 of each type), 4POis)', 'Coupling of 4 (8 rovers (4 of each type), 4POis)'], fontsize=20)


plt.show()



'''
import numpy as np
import matplotlib.pyplot as plt
import csv

x, y = [], []

base_folder = '/home/aadi-z640/research/CERL_results/'



with open(base_folder + 'Humanoid_1_learner/without_noise/score_Humanoid-v2_p10_s2018.csv','r') as csvfile:

	plots1 = csv.reader(csvfile, delimiter=',')

	for row in plots1:

		x.append(float(row[0]))
		y.append(float(row[1]))

plt.plot(x, y, color = 'blue', alpha = 1,label='5 learners (with noise)')

cumsum, moving_aves = [0], []
N=20

moving_aves = y[:N]
max = y[:N]
min = y[:N]
list_elements =y[:N]
max_elements, min_elements =  y[:N],y[:N]

for i, element in enumerate(y, 1):
	cumsum.append(cumsum[i-1] + element)
	list_elements.append(element)
	if i>N:
		moving_ave = (cumsum[i] - cumsum[i-N])/N
		#can do stuff with moving_ave here
		moving_aves.append(moving_ave)

		max_elements.append(np.max(list_elements))
		min_elements.append(np.max(list_elements))
		list_elements.pop(len(list_elements)-1)

plt.plot(x, moving_aves, color = 'red',  label='Average for 20 subsequent steps')

######################################

'''
'''
cumsum, moving_aves = [0], []
N=20

moving_aves = y[:N]
max = y[:N]
min = y[:N]
list_elements =y[:N]
max_elements, min_elements =  y[:N],y[:N]

for i, element in enumerate(y, 1):
	cumsum.append(cumsum[i-1] + element)
	list_elements.append(element)
	if i>N:
		moving_ave = (cumsum[i] - cumsum[i-N])/N
		#can do stuff with moving_ave here
		moving_aves.append(moving_ave)

		max_elements.append(np.max(list_elements))
		min_elements.append(np.max(list_elements))
		list_elements.pop(len(list_elements)-1)

plt.plot(x, moving_aves, color = 'orange',  label='Average for 20 subsequent steps')

'''

'''

plt.rcParams.update({'font.size': 30})

plt.tick_params(labelsize=30)

plt.xlabel('Time Steps')
plt.ylabel('Performance (Average Reward)', fontsize=30)
plt.title('Humanoid', fontsize=30)
plt.legend()

plt.show()
'''