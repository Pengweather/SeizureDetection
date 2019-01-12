import os

class Pairing:
	def __init__ (self, ChannelNo, PairingNo):
		Filepath = "channel" + str(ChannelNo) + "_pair" + str(PairingNo) + ".csv"
		print(Filepath)		
		if (os.path.exists(Filepath) == False):
			raise OSError("Pairing not found")
		self.ID = PairingNo
		self.data = []		
		with open(Filepath) as Fp:
			for line in Fp:
				self.data.append(float(line))

