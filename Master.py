import pickle
import numpy as np

def generateChannelMasterFile (filepath):
	with open(filepath + ".txt") as fp:
		studyNameAndChannelNo  = fp.readline().split(": ")[1]
		studyName = " ".join(studyNameAndChannelNo.split(" ", 2)[:2])
		channelNo = int(studyNameAndChannelNo.split(" ", 2)[2][1])
		filterOrder = int(fp.readline().split(": ")[1][0])
		filterCutoff_1 = float(fp.readline().split(": ")[1][:-1])
		filterCutoff_2 = float(fp.readline().split(": ")[1][:-1])
		Ts = float(fp.readline().split(": ")[1][:-1])
		Fs = float(fp.readline().split(": ")[1][:-1])
		dataLength = int(fp.readline().split(": ")[1][:-1])
		seizureLength = int(fp.readline().split(": ")[1][:-1])
		seizureDuration = float(fp.readline().split(": ")[1][:-1])

		seizureStart = []
		seizureEnd = [0]
		seizureDuration = []

		# Reading in Seizure Start
		for temp in range(3):
			next(fp)
		count = 0
		for line in fp:
			seizureStart.append(int(line[:-1]))
			count = count + 1
			if (count == seizureLength):
				break

		# Reading in Seizure End
		for temp in range(3):
			next(fp)
		count = 0
		for line in fp:
			seizureEnd.append(int(line[:-1]))
			count = count + 1
			if (count == seizureLength):
				break

		# Reading in Seizure Duration
		for temp in range(3):
			next(fp)
		count = 0
		for line in fp:
			seizureDuration.append(float(line[:-1]))
			count = count + 1
			if (count == seizureLength):
				break

	fp.close()
	relativeIdx = np.asarray(seizureStart) - np.asarray(seizureEnd[0:len(seizureEnd)-1])
	master = {"seizureStart": relativeIdx, "studyName": studyName, "channelNo": channelNo, "filterOrder": filterOrder, "filterCutoff_1": filterCutoff_1, "filterCutoff_2": filterCutoff_2, "Ts": Ts, "Fs": Fs, "dataLength": dataLength, "seizureDuration": np.asarray(seizureDuration), "seizureLength": seizureLength}
	pickle.dump(master, open(filepath + ".pkg","wb"))

generateChannelMasterFile('Study_005_channel1')

#data = pickle.load(open('Study_005_channel1.pkg',"rb"))
#print(data["seizureStart"][14])
