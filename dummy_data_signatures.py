from signal import signal
import pandas as pd
from datetime import datetime
import sys
import esig
import numpy as np

np.set_printoptions(threshold=sys.maxsize)  

#observed airports -> oa
oa = [np.int_(0), np.int_(1), np.int_(2)]
depth = 2
stream_length = 173000 # 2 days


relevant_events = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\data.csv')

relevant_events = relevant_events[relevant_events["icao"].isin(oa)]

split_stream = []
stream = []
entry = [0.0 for i in range(len(oa) * 2 + 1)]
end_timestamp = stream_length

for _, row in relevant_events.iterrows():
    if row["timestamp"] > end_timestamp:
        end_timestamp += stream_length
        split_stream.append(np.array(stream)) 
        stream = []
        entry = [0.0 for i in range(len(oa) * 2 + 1)]
    index = 0
    for i in range(len(oa)):
        if row['icao'] == oa[i]:
            index = 2 * i
    if row['event'] == 'departed':
        index += 1
    entry[index] += 1
    entry[-1] = row['timestamp']
    stream.append(entry.copy())

split_stream.append(np.array(stream)) 
signatures = []

for stream in split_stream:
    sig = esig.stream2sig(stream, depth)
    signatures.append(sig)

print(len(signatures))

df = pd.DataFrame(signatures)
df.to_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures.csv')
