import librosa as librosa
import math
import matplotlib.pyplot as plt
from librosa import display
import numpy as np

samples, sampling_rate = librosa.load('./music/faded.wav')

# i = 500000
# while i < 500050:
#     print(samples[i])
#     i += 1


print('sampling rate is: ' + str(sampling_rate))
print('sample count is: ' + str(len(samples)))
duration = len(samples) / sampling_rate
duration_floor = math.floor(len(samples) / sampling_rate/60)
seconds = duration - (duration_floor * 60)
seconds_floor = math.floor(seconds)
milliseconds = math.floor((seconds - (seconds_floor)) * 1000)
print('song length is: ' + str(duration_floor) + ':' + str(seconds_floor) + ':' + str(milliseconds))

plt.figure()
librosa.display.waveplot(y = samples, sr = sampling_rate)
plt.show()

def spectrogram(samples, sample_rate, stride_ms = 10.0, window_ms = 10.0, max_freq = 2.2*10**4, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram

data = spectrogram(samples, sampling_rate)



# plt.subplot(212)
plt.specgram(data, Fs = sampling_rate)
plt.show()
