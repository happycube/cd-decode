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
NYQUIST_MHZ = FREQ_MHZ / 2
FREQ_HZ = FREQ_MHZ * 1000000.0
NYQUIST_HZ = FREQ_HZ / 2

data = np.fromfile("ldd.raw", dtype = np.uint8)

# remove the first samples because they are strange (lower amplitude)
data = data[2650:len(data)-5000]

#for i in range(0, len(data)):
#	print i / FREQ_HZ,",", (data[i] / 256.0) - .5
#
#exit()

# without filter:  299/964
# poles at 0 and 49700 hz, 3.202312738us, zero at 1.59mhz/0.100097448us 

# this shhould be - but doesn't work worth a darn
deemp_pole = .100097448 * 1 
deemp_zero = 3.202312738 * 1 

# 21/1018
# NEW MEASURE:  850/1079
deemp_pole = .1100 * 1 
deemp_zero = 3.100 * 1 
lowpass_b, lowpass_a = sps.butter(2, 2.120/NYQUIST_MHZ)

# 999/1050 
deemp_pole = .1450 * 1 
deemp_zero = 3.100 * 1 
lowpass_b, lowpass_a = sps.butter(2, 2.120/NYQUIST_MHZ)

# 1177/798
deemp_pole = .1450 * 1 
deemp_zero = 3.100 * 1 
lowpass_b, lowpass_a = sps.butter(4, 2.200/NYQUIST_MHZ)

# 1204/771
deemp_pole = .1450 * 1 
deemp_zero = 3.202312738 * 1 
lowpass_b, lowpass_a = sps.butter(4, 2.200/NYQUIST_MHZ)

# 1204/771
deemp_pole = .1450 * 1 
deemp_zero = 3.202312738 * 1 
lowpass_b, lowpass_a = sps.butter(4, 2.200/NYQUIST_MHZ)

# LDD - 0 good 1593 bad 
deemp_pole = .1000 * 1 
deemp_zero = 3.202312738 * 1 
# below from megapixie
lowpass_b, lowpass_a = sps.cheby2(16, 100., 2.15 / NYQUIST_MHZ)  # Looks a bit odd, but is a reasonable tie for the spec filter (-26db at 2.0 Mhz, -50+ at 2.3Mhz)

# LDD - 0 good 2185bad 31046 symbols 
deemp_pole = .0400 * 1 
deemp_zero = 3.202312738 * 1 
# below from megapixie
lowpass_b, lowpass_a = sps.cheby2(16, 100., 2.15 / NYQUIST_MHZ)  # Looks a bit odd, but is a reasonable tie for the spec filter (-26db at 2.0 Mhz, -50+ at 2.3Mhz)

# LDD - 0/2521/40740 
deemp_pole = .0400 * 1 
deemp_zero = 3.202312738 * 1 
lowpass_b, lowpass_a = sps.cheby2(14, 100., 1.80 / NYQUIST_MHZ)  

# LDD - 0/2531/41244 
deemp_pole = .0370 * 1 
deemp_zero = 3.102312738 * 1 
lowpass_b, lowpass_a = sps.cheby2(14, 100., 1.80 / NYQUIST_MHZ)  

# LDD - 0/2535/41337 
deemp_pole = .0360 * 1 
deemp_zero = 3.102312738 * 1 
lowpass_b, lowpass_a = sps.cheby2(14, 100., 1.80 / NYQUIST_MHZ)  

# LDD - 0/2535/41337 
deemp_pole = .0360 * 1 
deemp_zero = 3.102312738 * 1 
lowpass_b, lowpass_a = sps.cheby2(14, 100., 1.800 / NYQUIST_MHZ)  

[tf_b, tf_a] = sps.zpk2tf([-deemp_pole*(10**-8)], [-deemp_zero*(10**-8)], deemp_zero / deemp_pole)
[f_emp_b, f_emp_a] = sps.bilinear(tf_b, tf_a, .5/FREQ_HZ)

# .295 leftover scale:  1053 frames found, 34731 good samps, 374 ERRORS
bandpass = sps.firwin(55, [.335/NYQUIST_MHZ, 1.870/NYQUIST_MHZ], pass_zero=False)

# .295 leftover scale:  1053 frames found, 34731 good samps, 374 ERRORS
# 1647/352
bandpass = sps.firwin(53, [.335/NYQUIST_MHZ, 1.870/NYQUIST_MHZ], pass_zero=False)

# 1651/348
bandpass = sps.firwin(53, [.355/NYQUIST_MHZ, 1.870/NYQUIST_MHZ], pass_zero=False)
# 1652/343
bandpass = sps.firwin(53, [.355/NYQUIST_MHZ, 2.800/NYQUIST_MHZ], pass_zero=False)
# 1660/341
bandpass = sps.firwin(39, [.600/NYQUIST_MHZ, 2.800/NYQUIST_MHZ], pass_zero=False)
# 1660/339
bandpass = sps.firwin(37, [.900/NYQUIST_MHZ, 2.800/NYQUIST_MHZ], pass_zero=False)

#doplot(f_emp_b, f_emp_a)
#doplot(bandpass, [1.0])
#exit()

# convert to single-precision floats
#data = data.astype(np.float32)

# fewer errors if we filter as double precision
data = data.astype(np.float64)

# subtract DC component
dc = data.mean()
data -= dc

#plt.plot(data[5000:6000])

data = sps.lfilter(lowpass_b, lowpass_a, data)
data = sps.lfilter(f_emp_b, f_emp_a, data)

#data = sps.lfilter(bandpass, [1.0], data)

plt.plot(data[5000:6000])
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
print "bias: {} seconds".format(bias), (totalRunlength0 - totalRunlength1) 

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
            #durationr = int(round(duration + (leftover * 0.22))) # to integer
            #durationr = int(round(duration + (leftover * 0.24))) # to integer
#            durationr = int(round(duration + (leftover * 0.270))) # to integer
            durationr = int(round(duration + (leftover * 0.295))) # to integer
#           durationr = int(round(duration)) # to integer
            leftover = duration - durationr

            f.write(str(value) * durationr)
