import Pairing
import pickle
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class Measurement:
	def __init__ (self, filepath):
		print("Entering")		
		# Parameters to extract from the text file and store into the variables below
		self.seizurePairings = []
		# Opening the text file here (For Debugging)
		# filepath = 'Study_005_channel1.txt'
		self.label = np.asarray([])
		self.seizureDuration = []
		self.seizureData = []
		# Retrieving all the data from the text file
		with open(filepath,"rb") as fp:
			# Reading in parameters
			master = pickle.load(open(filepath,"rb"))
			#StudyNameAndChannelNo  = fp.readline().split(": ")[1]
			self.StudyName = master["StudyName"]
			self.ChannelNo = int(master["ChannelNo"])
			self.FilterOrder = int(master["FilterOrder"])
			self.FilterCutoff_1 = float(master["FilterCutoff_1"])
			self.FilterCutoff_2 = float(master["FilterCutoff_2"])
			self.Ts = float(master["Ts"])
			self.Fs = float(master["Fs"])
			self.SeizureLength = int(master["SeizureLength"])
			
		fp.close()

		# Doing the pairings here
		for i in range(1, self.SeizureLength + 1):
			try:
				Pair = Pairing.Pairing(self.ChannelNo, i)
			except:
				print("Did not find Channel", self.ChannelNo, "Pairing", i)
			else:
				print("Found Channel", self.ChannelNo, "Pairing", i)
				temp_label = np.zeros(len(Pair.data))
				self.seizureData += (Pair.data)
				temp_start = master["seizureStart"][i-1]
				print(temp_start)
				temp_label[temp_start:] = 1
				self.label = np.append(self.label,temp_label)
				self.seizureDuration.append(master["SeizureDuration"][i-1])

		print(len(self.label))
		print(len(self.seizureData))

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

x = Measurement("Study_005_channel1.pkg")
# x.downsample(2)
#data = np.array(lines,np.float32)
#print("data length: ", len(data))
#plt.figure()
#plt.plot(data[0::2]**2)
#leg = plt.legend()
#plt.setp(leg.get_lines(),linewidth = 0.5)
#plt.show()
