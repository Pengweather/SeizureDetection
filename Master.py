import pickle
import numpy as np
def generateChannelMasterFile (filepath):
	with open(filepath+".txt") as fp:
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

	fp.close()
	Relative_idx = np.asarray(seizureStart) - np.asarray(seizureEnd[0:len(seizureEnd)-1])
	master = {"seizureStart": Relative_idx, "StudyName": StudyName, "ChannelNo": ChannelNo, "FilterOrder": FilterOrder, "FilterCutoff_1": FilterCutoff_1, "FilterCutoff_2": FilterCutoff_2, "Ts": Ts, "Fs": Fs, "DataLength": DataLength, "SeizureDuration": np.asarray(seizureDuration), "SeizureLength": SeizureLength}
	#print(master)
	#print(master["SeizureDuration"])
	pickle.dump(master, open(filepath + ".pkg","wb"))

generateChannelMasterFile('Study_005_channel1')
#data = pickle.load(open('Study_005_channel1.pkg',"rb"))
#print(data["seizureStart"][14])
