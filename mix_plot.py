import matplotlib.pyplot as plt
import numpy as np
import os

SAMPLING_DIR = './mix_sample/'
data_file_dir = os.listdir(SAMPLING_DIR)
SAVE = 1


data_list = []
for data_file in data_file_dir:
	data_path = SAMPLING_DIR + data_file
	data = []
	with open(data_path, 'rb') as f:
		for line in f:
			parse = [float(line.rstrip('\n'))]
			data.append(parse)
	file_name = data_file.split('.')[0]
	data_list.append([file_name, data])

figs = []
for data in data_list:
	p = plt.figure(figsize=(20,5))
	plt.plot([float(x)/2 for x in range(0,len(data[1]))], data[1], color='chocolate', label=data[0], linewidth=1.5,alpha=0.9)
	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, len(data[1])/2, 0, 1200])
	plt.xticks(np.arange(0, len(data[1])/2+1, 50))
	plt.yticks(np.arange(200, 1200+1, 200))
	figs.append([data[0], p])
	if SAVE:
		p.savefig('./mix_figures/' + data[0] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
	plt.close()

	# formated = [round(x,2) for x in j]
