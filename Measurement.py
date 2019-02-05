import Pairing
import Feature
import NonlinearEnergy
import pickle
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class Measurement:
	def __init__ (self, filepath, startPairing, endPairing):
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
			self.studyName = master["studyName"]
			self.channelNo = int(master["channelNo"])
			self.filterOrder = int(master["filterOrder"])
			self.filterCutoff_1 = float(master["filterCutoff_1"])
			self.filterCutoff_2 = float(master["filterCutoff_2"])
			self.Ts = float(master["Ts"])
			self.Fs = float(master["Fs"])
			self.seizureLength = int(master["seizureLength"])		
		fp.close()

		# Doing the pairings here
		for i in range(startPairing, endPairing + 1):
			try:
				Pair = Pairing.Pairing(self.channelNo, i)
			except:
				print("Did not find Channel", self.channelNo, "Pairing", i)
			else:
				# print("Found Channel", self.channelNo, "Pairing", i)
				temp_label = np.zeros(len(Pair.data))
				self.seizureData += (Pair.data)
				temp_start = master["seizureStart"][i-1]
				temp_label[temp_start:] = 1
				self.label = np.append(self.label,temp_label)
				self.seizureDuration.append(master["seizureDuration"][i-1])

		assert(len(self.label) == len(self.seizureData))
		

	def downsample(self, n = 2):
		if (self.Fs % n != 0):
			raise FrequencyIntegerError("Resulting sampling frequency must be an integer")
		# Decimate
		self.seizureData = signal.decimate(np.array(self.seizureData), n)
		# Updating the length, sampling frequency, and sampling time
		self.dataLength = self.seizureData.size
		self.Fs = self.Fs/n
		self.Ts = 1/self.Fs
		# Updating the seizureStart and seizureEnd
		# self.seizureStart = np.floor(np.array(self.seizureStart)/n).astype(int)
		# self.seizureEnd = np.ceil(np.array(self.seizureEnd)/n).astype(int)
		# Updating the seizurePairings
		self.label = self.label[::n]
