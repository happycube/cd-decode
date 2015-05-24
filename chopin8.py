#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import itemfreq

import numpy as np
import scipy.signal as sps

freq = (315.0 / 88.0) * 8.0

def doplot(B, A):
	w, h = sps.freqz(B, A)

	fig = plt.figure()
	plt.title('Digital filter frequency response')
	
	db = 20 * np.log10(abs(h))

	ax1 = fig.add_subplot(111)
	
	plt.plot(w * (freq/np.pi) / 2.0, 20 * np.log10(abs(h)), 'b')
	plt.ylabel('Amplitude [dB]', color='b')
	plt.xlabel('Frequency [rad/sample]')

	ax2 = ax1.twinx()
	angles = np.unwrap(np.angle(h))
	plt.plot(w * (freq/np.pi) / 2.0, angles, 'g')
	plt.ylabel('Angle (radians)', color='g')
	
	plt.grid()
	plt.axis('tight')
	plt.show()


CD_BASE_FREQUENCY = 4321800.0 # Hz
SAMPLE_FREQUENCY = 28.636e6 # Hz

FREQ_MHZ = (315.0 / 88.0) * 8.0
FREQ_HZ = FREQ_MHZ * 1000000.0
NYQUIST_MHZ = FREQ_MHZ / 2

data = np.fromfile("chopin8.cdraw", dtype = np.uint8)

# remove the first samples because they are strange (lower amplitude)
data = data[2650:]

# convert to single-precision floats
#data = data.astype(np.float32)

# fewer errors if we filter as double precision
data = data.astype(np.float64)

# subtract DC component
dc = data.mean()
data -= dc

# without filter: 340 errors, 303bad/449good if DP (324/473 if using leftover at write time)

# 91 errors
bandpass = sps.firwin(97, [.08/NYQUIST_MHZ, 1.20/NYQUIST_MHZ], pass_zero=False)
# 88 errors
bandpass = sps.firwin(97, [.075/NYQUIST_MHZ, 1.20/NYQUIST_MHZ], pass_zero=False)
# 66 errors, 53 if double precision 
bandpass = sps.firwin(97, [.075/NYQUIST_MHZ, 1.50/NYQUIST_MHZ], pass_zero=False)
# 47 (double precision)
bandpass = sps.firwin(97, [.100/NYQUIST_MHZ, 1.50/NYQUIST_MHZ], pass_zero=False)
# 44 (double precision)
bandpass = sps.firwin(91, [.100/NYQUIST_MHZ, 1.50/NYQUIST_MHZ], pass_zero=False)
# 40 (double precision)/ 842 good - 29/920 if using leftover
bandpass = sps.firwin(91, [.095/NYQUIST_MHZ, 1.72/NYQUIST_MHZ], pass_zero=False)

# 30/957 with leftover
bandpass = sps.firwin(65, [.150/NYQUIST_MHZ, 1.75/NYQUIST_MHZ], pass_zero=False)

# 20/994 with leftover
bandpass = sps.firwin(49, [.290/NYQUIST_MHZ, 1.80/NYQUIST_MHZ], pass_zero=False)

# 17/999 with leftover
bandpass = sps.firwin(49, [.290/NYQUIST_MHZ, 1.85/NYQUIST_MHZ], pass_zero=False)

# 19/1007 with leftover
bandpass = sps.firwin(45, [.360/NYQUIST_MHZ, 1.85/NYQUIST_MHZ], pass_zero=False)
# 18/1008 with leftover, 36/1055 with new leftover
#bandpass = sps.firwin(47, [.350/NYQUIST_MHZ, 1.85/NYQUIST_MHZ], pass_zero=False)

# 8/1054 with 0.2 leftover
bandpass = sps.firwin(55, [.350/NYQUIST_MHZ, 1.85/NYQUIST_MHZ], pass_zero=False)

# trying to follow cd player as reference - 79/825
#bandpass = sps.firwin(33, [.047/NYQUIST_MHZ, 1.59/NYQUIST_MHZ], pass_zero=False)

data = sps.lfilter(bandpass, 1.0, data)

#[b, a] = sps.zpk2tf([1/1590000.0], [0, 1/49700.0], 1.0)
#[bb, aa] = sps.bilinear(b, a, 1.0)
#doplot(bb, aa)
#exit()

bandpassb, bandpassa = sps.butter(8, [0.047/NYQUIST_MHZ, 1.59/NYQUIST_MHZ], btype='bandpass')
#doplot(bandpassb, bandpassa)
#exit()
#plt.plot(data[5000:6000])
#data = sps.lfilter(bandpassb, bandpassa, data)

#plt.plot(data[5000:6000])
#plt.show()
#exit()

# filter to binary signal
data = (data > 0.0)

transition = np.diff(data) != 0
transition = np.insert(transition, 0, False) # The first sample is never a transition.

print "data", data.shape, data.dtype
print "transition", transition.shape, transition.dtype

runLengths = np.diff(np.where(transition)[0])

# fetch run signal values. The last transition
# isn't part of a well-defined run, so we don't need it.

runValues = data[transition].astype(np.int8)
runValues = runValues[:-1]

print "runLengths", runLengths.shape, runLengths.dtype
print "runValues", runValues.shape, runValues.dtype

totalRunlength0 = np.sum(runLengths[runValues == 0])
totalRunlength1 = np.sum(runLengths[runValues == 1])

bias = (totalRunlength0 - totalRunlength1) / (SAMPLE_FREQUENCY * len(runLengths))
print "bias: {} seconds".format(bias)

runDurations = runLengths / SAMPLE_FREQUENCY   # to SECONDS

runDurations[runValues == 0] -= bias
runDurations[runValues == 1] += bias

runDurations = runDurations * CD_BASE_FREQUENCY # to CD BASE FREQUENCY TICKS

if False:

    print "plotting ..."

    freqAll = itemfreq(runDurations)
    freq0 = itemfreq(runDurations[runValues == 0])
    freq1 = itemfreq(runDurations[runValues == 1])

    plt.subplot(411)
    plt.title("All runs (bias corrected)")
    plt.xlim(0, 13)
    plt.plot(freqAll[:, 0], freqAll[:, 1], '.-')

    plt.subplot(412)
    plt.title("Bias-corrected zero runs ")
    plt.xlim(0, 13)
    plt.plot(freq0[:, 0], freq0[:, 1], '.-')

    plt.subplot(413)
    plt.title("Bias-corrected one runs")
    plt.xlim(0, 13)
    plt.plot(freq1[:, 0], freq1[:, 1], '.-')

    plt.subplot(414)
    plt.title("Bias-corrected zero/one runs, overlayed")
    plt.xlim(0, 13)
    plt.plot(freq0[:, 0], freq0[:, 1], '.-')
    plt.plot(freq1[:, 0], freq1[:, 1], '.-')

    plt.savefig("chopin8.pdf")
    plt.close()

if True:

    print "writing file ..."

    with open("chopin8-bits.txt", "w") as f:
        leftover = 0
        for (value, duration) in zip(runValues, runDurations):
            #durationr = int(round(duration + (leftover * .111))) # to integer
            durationr = int(round(duration + (leftover * 0.2))) # to integer
#           durationr = int(round(duration)) # to integer
            leftover = duration - durationr

            f.write(str(value) * durationr)
