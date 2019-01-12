import Pairing
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class Measurement:
	def __init__ (self, filepath):
		print("Entering")		
		# Parameters to extract from the text file and store into the variables below
		self.StudyName = ""
		self.ChannelNo = 0
		self.FilterOrder = 0
		self.FilterCutoff_1 = 0.0
		self.FilterCutoff_2 = 0.0
		self.Ts = 0.0
		self.Fs = 0.0
		self.DataLength = 0
		self.SeizureLength = 0
		self.SeizureDuration = 0.0

		# Data arrays
		self.seizureStart = []
		self.seizureEnd = []
		self.seizureDuration = []
		self.seizureData = []
		self.seizurePairings = []

		# Opening the text file here (For Debugging)
		# filepath = 'Study_005_channel1.txt'

		# Retrieving all the data from the text file
		with open(filepath) as fp:
			# Reading in parameters
			StudyNameAndChannelNo  = fp.readline().split(": ")[1]
			self.StudyName = " ".join(StudyNameAndChannelNo.split(" ", 2)[:2])
			self.ChannelNo = int(StudyNameAndChannelNo.split(" ", 2)[2][1])
			self.FilterOrder = int(fp.readline().split(": ")[1][0])
			self.FilterCutoff_1 = float(fp.readline().split(": ")[1][:-1])
			self.FilterCutoff_2 = float(fp.readline().split(": ")[1][:-1])
			self.Ts = float(fp.readline().split(": ")[1][:-1])
			self.Fs = float(fp.readline().split(": ")[1][:-1])
			self.DataLength = int(fp.readline().split(": ")[1][:-1])
			self.SeizureLength = int(fp.readline().split(": ")[1][:-1])
			self.SeizureDuration = float(fp.readline().split(": ")[1][:-1])

			# Reading in Seizure Start
			for temp in range(3):
				next(fp)
			count = 0
			for line in fp:
				self.seizureStart.append(int(line[:-1]))
				count = count + 1
				if (count == self.SeizureLength):
					break

			# Reading in Seizure End
			for temp in range(3):
				next(fp)
			count = 0
			for line in fp:
				self.seizureEnd.append(int(line[:-1]))
				count = count + 1
				if (count == self.SeizureLength):
					break

			# Reading in Seizure Duration
			for temp in range(3):
				next(fp)
			count = 0
			for line in fp:
				self.seizureDuration.append(float(line[:-1]))
				count = count + 1
				if (count == self.SeizureLength):
					break

			# Reading in Seizure Data
			for temp in range(3):
				next(fp)
			for line in fp:
				self.seizureData.append(float(line[:-1]))
		fp.close()

		# Doing the pairings here
		for i in range(1, self.SeizureLength + 1):
			try:
				Pair = Pairing.Pairing(self.ChannelNo, i)
			except:
				print("Did not find Channel", self.ChannelNo, "Pairing", i)
			else:
				print("Found Channel", self.ChannelNo, "Pairing", i)
				self.seizurePairings.append(Pair)

	def downsample(self, n = 2):
		if (self.Fs % n != 0):
			raise FrequencyIntegerError("Resulting sampling frequency must be an integer")
		# Decimate
		self.seizureData = signal.decimate(np.array(self.seizureData), n)
		# Updating the length, sampling frequency, and sampling time
		self.DataLength = self.seizureData.size
		self.Fs = self.Fs/n
		self.Ts = 1/self.Fs
		# Updating the seizureStart and seizureEnd
		self.seizureStart = np.floor(np.array(self.seizureStart)/n).astype(int)
		self.seizureEnd = np.ceil(np.array(self.seizureEnd)/n).astype(int)
		# Updating the seizurePairings
		for i in range(0, len(self.seizurePairings)):
			print(len(self.seizurePairings[i].data))
			self.seizurePairings[i].data = self.seizurePairings[i].data[::n]
			print(len(self.seizurePairings[i].data))

x = Measurement("Study_005_channel1.txt")
x.downsample(2)
#data = np.array(lines,np.float32)
#print("data length: ", len(data))
#plt.figure()
#plt.plot(data[0::2]**2)
#leg = plt.legend()
#plt.setp(leg.get_lines(),linewidth = 0.5)
#plt.show()
