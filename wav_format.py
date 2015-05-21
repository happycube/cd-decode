# wav_format.py
#
# Copyright (c) 2005 by Sidney Cadot <sidney@jigsaw.nl>
# This software is licensed under the GNU General Public License (GPL).
#
# This file is part of laser2wav, a software-only implementation of
# an audio CD decoder.
#
###############################################################################

def little_endian_long(n):
    return [(n)&255, (n>>8)&255, (n>>16)&255, (n>>24)&255]

def little_endian_short(n):
    return [(n)&255, (n>>8)&255]

def clip(n):
    if (n<-32768): n = -32768
    if (n>32767): n = 32767
    return n

def to_unsigned(n):
    if (n<0):
        n = n + 65536
    return n

def wave_file_string(audio_samples):

    bytes = []

    num_samples     = len(audio_samples)
    subchunk1size   = 16
    audio_format    = 1
    num_channels    = 2
    sample_rate     = 44100
    bits_per_sample = 16
    byte_rate       = sample_rate  * num_channels * bits_per_sample/8
    block_align     = num_channels * bits_per_sample/8
    subchunk2size   = num_samples  * num_channels * bits_per_sample/8

    chunksize       = 4 + (8+subchunk1size) + (8+subchunk2size)

    bytes.extend([ord(x) for x in "RIFF"])
    bytes.extend(little_endian_long(chunksize))
    bytes.extend([ord(x) for x in "WAVE"])
    bytes.extend([ord(x) for x in "fmt "])
    bytes.extend(little_endian_long(subchunk1size))
    bytes.extend(little_endian_short(audio_format))
    bytes.extend(little_endian_short(num_channels))
    bytes.extend(little_endian_long(sample_rate))
    bytes.extend(little_endian_long(byte_rate))
    bytes.extend(little_endian_short(block_align))
    bytes.extend(little_endian_short(bits_per_sample))
    bytes.extend([ord(x) for x in "data"])
    bytes.extend(little_endian_long(subchunk2size))

    for (L, R) in audio_samples:
        bytes.extend(little_endian_short(to_unsigned(clip(L))))
        bytes.extend(little_endian_short(to_unsigned(clip(R))))

    return str.join("", map(chr, bytes))
