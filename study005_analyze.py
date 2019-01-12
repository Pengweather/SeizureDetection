import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

# Parameters to extract from the text file and store into the variables below
StudyName = ""
ChannelNo = 0
FilterOrder = 0
FilterCutoff_1 = 0.0
FilterCutoff_2 = 0.0
Ts = 0.0
Fs = 0.0
DataLength = 0
SeizureLength = 0
SeizureDuration = 0.0

# Data arrays
seizureStart = []
seizureEnd = []
seizureDuration = []
seizureData = []

# Opening the text file here
filepath = 'Study_005_channel1.txt'

# Retrieving all the data from the text file
with open(filepath) as fp:
	# Reading in parameters
	StudyNameAndChannelNo  = fp.readline().split(": ")[1]
	StudyName = " ".join(StudyNameAndChannelNo.split(" ", 2)[:2])
	ChannelNo = int(StudyNameAndChannelNo.split(" ", 2)[2][1])
	FilterOrder = int(fp.readline().split(": ")[1][0])
	FilterCutoff_1 = float(fp.readline().split(": ")[1][:-1])
	FilterCutoff_2 = float(fp.readline().split(": ")[1][:-1])
	Ts = float(fp.readline().split(": ")[1][:-1])
	Fs = float(fp.readline().split(": ")[1][:-1])
	DataLength = int(fp.readline().split(": ")[1][:-1])
	SeizureLength = int(fp.readline().split(": ")[1][:-1])
	SeizureDuration = float(fp.readline().split(": ")[1][:-1])

	# Reading in Seizure Start
	for temp in range(3):
		next(fp)
	count = 0
	for line in fp:
		seizureStart.append(int(line[:-1]))
		count = count + 1		
		if (count == SeizureLength):
			break

	# Reading in Seizure End
	for temp in range(3):
		next(fp)
	count = 0
	for line in fp:
		seizureEnd.append(int(line[:-1]))
		count = count + 1
		if (count == SeizureLength):
			break

	# Reading in Seizure Duration
	for temp in range(3):
		next(fp)
	count = 0
	for line in fp:
		seizureDuration.append(float(line[:-1]))
		count = count + 1		
		if (count == SeizureLength):
			break

	# Reading in Seizure Data
	for temp in range(3):
		next(fp)
	for line in fp:
		seizureData.append(float(line[:-1]))	

fp.close()

"""
print(seizureData)
print(StudyName)
print(ChannelNo)
print(FilterOrder)
print(FilterCutoff_1)
print(FilterCutoff_2)
print(Ts)
print(Fs)
print(DataLength)
print(SeizureLength)
print(SeizureDuration)
"""

print(seizureStart)
labels = np.zeros(DataLength)
for i in range(SeizureLength):
    labels[seizureStart[i]:seizureEnd[i]] = 1;

seizureData = np.array(seizureData)

mpl.use('Agg')
plt.figure()
plt.plot(seizureData[3500000:3850000:10])
plt.plot(labels[3500000:3850000:10])
plt.show()
plt.savefig('Study_005_channel1_Seizure_Data.png')

#data = np.array(lines,np.float32)
#print("data length: ", len(data))
#plt.figure()
#plt.plot(data[0::2]**2)
#leg = plt.legend()
#plt.setp(leg.get_lines(),linewidth = 0.5)
#plt.show()
