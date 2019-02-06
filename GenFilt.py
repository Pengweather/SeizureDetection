import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

def filt_data(data, low = 14, high = 32, Fs = 500, ntap = 10, type = 'band'):
	sos = gen_filt_sos(low, high, Fs, ntap, type)
	result = signal.sosfilt(sos, data)
	return result

def show_resp(w,h):
    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.abs(h))
    plt.plot(w/np.pi*250, db)
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.subplot(2, 1, 2)
    plt.plot(w/np.pi, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    plt.show()

def gen_filt_sos(low, high, Fs, ntap, type):
    nyq = Fs/2
    cutoffs = [low/nyq,high/nyq]
    z,p,k = signal.butter(ntap, cutoffs, btype = type, analog = False, output='zpk')
    sos = signal.zpk2sos(z, p, k)
    return sos

# def test(low = 14.0, high = 32.0, Fs = 500, ntap = 10):
#     sos = gen_filt_sos(low, high, Fs, ntap, 'band')
#     w, h = signal.sosfreqz(sos, worN=1500)
# 	show_resp(w,h)
# b,a = gen_filt_sos(14.0, 32.0, 500, 10, 'band')
# w, h = signal.freqz(b,a, worN=1500)
# show_resp(w,h)
