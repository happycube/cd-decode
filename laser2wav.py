# laser2wav.py
#
# Copyright (c) 2005 by Sidney Cadot <sidney@jigsaw.nl>
# This software is licensed under the GNU General Public License (GPL).
#
# This file is part of laser2wav, a software-only implementation of
# an audio CD decoder.
#
###############################################################################

import sys
import wav_format
import galois_field as gf

from efm import EFM

def analyze_frame(frame):
    assert len(frame)==588 # verify that each frame is 588 bits long
    sync24 = frame[0:24]
    merge3 = frame[24:27] # ignore
    assert sync24 == "100000000001000000000010"

    framedata = []
    for i in range(33):
        channel14 = frame[27+17*i:41+17*i]
        merge3    = frame[41+17*i:44+17*i] # ignore
        data8 = EFM[channel14]
        framedata.append(data8)

    return (framedata[0], framedata[1:])

def invert_last_16(Q):
    return Q[:-16] + str.join("", map(lambda x: str(1-int(x)), Q[-16:]))

def valid_crc(Q):
    # this is a simple way to check the CRC
    if len(Q)==0: return True
    if Q[0]=='0': return valid_crc(Q[1:])
    if len(Q)<17: return False
    # subtract multiple of the generator polynomial
    Q = map(int, Q)
    Q[ 0] = 1-Q[ 0]
    Q[ 4] = 1-Q[ 4]
    Q[11] = 1-Q[11]
    Q[16] = 1-Q[16]
    Q = str.join("", map(str, Q))
    return valid_crc(Q)

def to_signed(n):
    if n<=32767:
        return n
    else:
        return n-65536

def bcd2int(S):
    assert len(S)==8
    hi = int(S[0:4], 2)
    #assert (hi>=0) and (hi<=9)
    lo = int(S[4:8], 2)
    #assert (lo>=0) and (lo<=9)
    return hi*10+lo

def analyze_control_stream(control_stream):

    print "Analyzing control stream ..."

    all_zeros = 96 * "0"
    all_ones  = 96 * "1"
    ok_map = {False: "ERROR", True: "ok"}

    while len(control_stream)>=98:
        if control_stream[0]!='SYNC0':
            # discard a control byte - not part of a sector
            control_stream = control_stream[1:]
        else:
            assert control_stream[0]=='SYNC0'
            assert control_stream[1]=='SYNC1'
            control_sector = control_stream[2:98]
            control_stream = control_stream[98:]

            P = str.join("", map(lambda x: str(int((x & 0x80)!=0)), control_sector))
            Q = str.join("", map(lambda x: str(int((x & 0x40)!=0)), control_sector))
            R = str.join("", map(lambda x: str(int((x & 0x20)!=0)), control_sector))
            S = str.join("", map(lambda x: str(int((x & 0x10)!=0)), control_sector))
            T = str.join("", map(lambda x: str(int((x & 0x08)!=0)), control_sector))
            U = str.join("", map(lambda x: str(int((x & 0x04)!=0)), control_sector))
            V = str.join("", map(lambda x: str(int((x & 0x02)!=0)), control_sector))
            W = str.join("", map(lambda x: str(int((x & 0x01)!=0)), control_sector))

            p_ok = (P in [all_zeros, all_ones])
            q_ok = valid_crc(invert_last_16(Q))
            r_ok = (R == all_zeros) # for most CDs
            s_ok = (S == all_zeros) # for most CDs
            t_ok = (T == all_zeros) # for most CDs
            u_ok = (U == all_zeros) # for most CDs
            v_ok = (V == all_zeros) # for most CDs
            w_ok = (W == all_zeros) # for most CDs

            print "    Sector sub-channels:"
            print "        P: <%s> %s" % (P, ok_map[p_ok])
            print "        Q: <%s> %s" % (Q, ok_map[q_ok])
            
            if q_ok:
                q_control = Q[0:4]
                q_mode    = int(Q[4:8],2)
                q_data    = Q[8:80]
                if q_mode == 1:
                    (tno,index,min,sec,frac,zero,amin,asec,afrac) = [bcd2int(q_data[8*i:8*i+8]) for i in range(9)]
                    assert zero == 0
                    print "              CONTROL .......... : <%s> [#channels / reserved / copy-protect / pre-emphasis]" % q_control
                    print "              MODE ............. : %d"       % q_mode
                    print "              DATA ............. : trackno. %02d indexno. %02d track-time %02d:%02d.%02d [mm:ss.ff] disc-time %02d:%02d.%02d [mm:ss.ff]" % (tno, index, min,  sec,  frac, amin, asec, afrac)
                else:
                    print "              CONTROL .......... : <%s>"  % q_control
                    print "              MODE ............. : %d"    % q_mode
                    print "              DATA ............. : <%s>"  % q_data

            print "        R: <%s> %s" % (R, ok_map[r_ok])
            print "        S: <%s> %s" % (S, ok_map[s_ok])
            print "        T: <%s> %s" % (T, ok_map[t_ok])
            print "        U: <%s> %s" % (U, ok_map[u_ok])
            print "        V: <%s> %s" % (V, ok_map[v_ok])
            print "        W: <%s> %s" % (W, ok_map[w_ok])

def verify_data_stream(data):

    errorsP = 0
    errorsQ = 0
    
    # make the Reed-Solomon check calculations

    invert = [0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255]

    # check C1 a.k.a. P-parity
    print "Checking C1 / P parity ..."
    p_delay = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    for i in range(1, len(data)):
        z_list = []
        for hp_row in range(4):
            z = gf.zero
            for j in range(32):
                z = gf.add(z, gf.multiply(gf.power(gf.alpha, hp_row*(31-j)), data[i-p_delay[j]][j]^invert[j]))
            z_list.append(z)    
        if z_list != [gf.zero, gf.zero, gf.zero, gf.zero]:
            print "P PARITY: ERROR DETECTED AT FRAME #%d" % i
            errorsP = errorsP + 1

    # check C2 a.k.a. Q-parity
    print "Checking C2 / Q parity ..."
    q_delay = [107,104,99,96,91,88,83,80,75,72,67,64,59,56,51,48,43,40,35,32,27,24,19,16,11,8,3,0]
    for i in range(107, len(data)):
        z_list = []
        for hq_row in range(4):
            z = gf.zero    
            for j in range(28):
                z  = gf.add(z, gf.multiply(gf.power(gf.alpha, hq_row*(27-j)), data[i-q_delay[j]][j]^invert[j]))
            z_list.append(z)    
        if z_list!= [gf.zero, gf.zero, gf.zero, gf.zero]:
            print "Q PARITY: ERROR DETECTED AT FRAME #%d" % i
            errorsQ = errorsQ + 1

    return (errorsP, errorsQ)

def extract_audio_stream(data):

    # extract the audio data

    print "Extracting audio data ..."
    audio_data = []

    for i in range(105, len(data)):

        L0  = to_signed(data[i-105][ 0]*256 + data[i-102][ 1])
        L2  = to_signed(data[i- 97][ 2]*256 + data[i- 94][ 3])
        L4  = to_signed(data[i- 89][ 4]*256 + data[i- 86][ 5])

        R0  = to_signed(data[i- 81][ 6]*256 + data[i- 78][ 7])
        R2  = to_signed(data[i- 73][ 8]*256 + data[i- 70][ 9])
        R4  = to_signed(data[i- 65][10]*256 + data[i- 62][11])

        L1  = to_signed(data[i- 43][16]*256 + data[i- 40][17])
        L3  = to_signed(data[i- 35][18]*256 + data[i- 32][19])
        L5  = to_signed(data[i- 27][20]*256 + data[i- 24][21])

        R1  = to_signed(data[i- 19][22]*256 + data[i- 16][23])
        R3  = to_signed(data[i- 11][24]*256 + data[i-  8][25])
        R5  = to_signed(data[i-  3][26]*256 + data[i-  0][27])

        audio_data.extend([(L0, R0), (L1, R1), (L2, R2), (L3, R3), (L4, R4), (L5, R5)])

    return audio_data

def main():

    if len(sys.argv)!=3:
        print "Usage: python laser2wav.py <laser-in.dat> <waveform-out.wav>"
        return

    filename_in  = sys.argv[1]
    filename_out = sys.argv[2]

    print "Reading raw signal ..."
    f = open(filename_in, "rb")
    raw_signal = f.read()
    f.close()
    
    print "Converting to delta signal..."
    raw_signal = map(int, raw_signal)
    delta_signal = [str(raw_signal[i] ^ raw_signal[i-1]) for i in range(1, len(raw_signal))]
    delta_signal = str.join("", delta_signal)

    print "Walking frames..."
    control_stream = []
    data_stream    = []

    z = 0
    while True:
        z = delta_signal.find("100000000001000000000010", z)
        if z<0: break
        frame = delta_signal[z:z+588]
        if len(frame)==588:
            (control, data) = analyze_frame(frame)
            control_stream.append(control)
            data_stream.append(data)
        z = z + 588

    analyze_control_stream(control_stream)

    verify_data_stream(data_stream)

    audio_data = extract_audio_stream(data_stream)

    print "Writing WAV file ..."
    f = open(filename_out, "wb")
    f.write(wav_format.wave_file_string(audio_data))
    f.close()

if __name__ == "__main__":
    main()
