import librosa as librosa
import math

samples, sampling_rate = librosa.load('./music/peer-gynt.wav')

i = 500000
while i < 500050:
    print(samples[i])
    i += 1
print('sampling rate is: ' + str(sampling_rate))
print('sample count is: ' + str(len(samples)))
duration = len(samples) / sampling_rate
duration_floor = math.floor(len(samples) / sampling_rate/60)
seconds = duration - (duration_floor * 60)
seconds_floor = math.floor(seconds)
milliseconds = math.floor((seconds - (seconds_floor)) * 1000)
print('song length is: ' + str(duration_floor) + ':' + str(seconds_floor) + ':' + str(milliseconds))