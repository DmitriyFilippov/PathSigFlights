import pandas as pd 
import numpy as np 
import esig 
import random

hour = 3600
day = 24 * hour

#distances between airports in time
times = [
[0, 3*hour, 0, 0, 4*hour],
[3*hour, 0, 6*hour, 4*hour, 0],
[0, 6*hour ,0, 0, 0],
[0, 4*hour, 0, 0, 7*hour],
[4*hour, 0, 0, 7*hour, 0]
]

file_location = ''
#edges present in the graph
edges = [(0,1), (0,4), (1, 0), (1, 2), (1, 3), (2, 1), (3, 1), (3, 4), (4, 0), (4, 3)]

#weights for regulating edge relative frequency in 2nd type of anomaly
rate_anomaly_weights = (1.5, 0.5, 1.5, 1, 0.5, 1, 0.5, 1.5, 0.5, 1.5)
fast_intervals_anomaly_wights = (1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0)

#delay between chain flights in 1st kind of anomaly
anomaly_chaining_delay_left = 10800
anomaly_chaining_delay_right = 12000

#random variation in flight length
flight_len_spread = 0.2


def normal(x, timestamp,id, data, anomaly_labels):
    for i in range(x):
        timestamp += (1800 + random.randrange(-800, 800))
        (a, b) = random.choice(edges)
        delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
        data.append([a, timestamp, "takeoff", id])
        data.append([b, timestamp+delay, "landed", id])
        id+= 1
    return (timestamp, id)

def chaining(x, timestamp, id, data, anomaly_labels):
    for i in range(6000):
        timestamp += (1800 + random.randrange(-800, 800))
        (a, b) = random.choice([x for x in edges if x != (1, 2)])
        delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
        data.append([a, timestamp, "takeoff", id])
        data.append([b, timestamp+delay, "landed", id])
        id += 1
        if(a == 0 and b == 1):                     #chaining only applies to this pair
            delay2 = random.randrange(anomaly_chaining_delay_left, anomaly_chaining_delay_right)               #delay between chained flights
            delay3 = 6 * 3600 * (random.random() * flight_len_spread + (1 - flight_len_spread/2))              #length of a second flight
            data.append([1, timestamp + delay + delay2, "takeoff", id])
            anomaly_labels.append([id, 1])
            data.append([2, timestamp + delay + delay2 + delay3, "landed", id])
            anomaly_labels.append([id, 1])
            id += 1
    
    return (timestamp, id)

def rate(x, timestamp,id, data, anomaly_labels):
    for i in range(x):
        timestamp += (1800 + random.randrange(-800, 800))
        (a, b) = random.choices(edges, weights = rate_anomaly_weights)[0]
        delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
        data.append([a, timestamp, "takeoff", id])
        anomaly_labels.append([id, 2])
        data.append([b, timestamp+delay, "landed", id])
        anomaly_labels.append([id, 2])
        id += 1
    
    return (timestamp, id)

def fast_intervals(x, timestamp,id, data, anomaly_labels):
    for i in range(x):
        timestamp += (1800 + random.randrange(-800,800))
        (a, b) = random.choices(edges, weights = fast_intervals_anomaly_wights)[0]
        delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
        data.append([a, timestamp, "takeoff", id])
        anomaly_labels.append([id, 3])
        data.append([b, timestamp+delay, "landed", id])
        anomaly_labels.append([id, 3])
        id += 1
        if a == 2:
            timestamp_ = timestamp
            for i in range(4):
                timestamp_ += (1800 + random.randrange(-800, 800))
                delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
                data.append([a, timestamp_, "takeoff", id])
                anomaly_labels.append([id, 3])
                data.append([b, timestamp_+delay, "landed", id])
                anomaly_labels.append([id, 3])
                id += 1
    
    return (timestamp, id)

def generate_data():
    data = []
    clean_data = []
    #timestamp is the running record of current timestamp
    timestamp = 0
    timestamp_clean = 0
    anomaly_labels = []
    id = 0

    clean_number = 100000
    anomaly_number = 10000

    timestamp, id = normal(clean_number * 3, timestamp, id, clean_data, anomaly_labels)

    timestamp, id = normal(clean_number, timestamp, id, data, anomaly_labels)
    
    timestamp, id = chaining(anomaly_number, timestamp, id, data, anomaly_labels)
    timestamp, id = normal(clean_number, timestamp, id, data, anomaly_labels)

    timestamp, id = rate(anomaly_number, timestamp, id, data, anomaly_labels)
    timestamp, id = normal(clean_number, timestamp, id, data, anomaly_labels)

    timestamp, id = fast_intervals(anomaly_number, timestamp, id, data, anomaly_labels)
    timestamp, id = normal(clean_number, timestamp, id, data, anomaly_labels)

    df = pd.DataFrame(data, columns = ["icao", "timestamp", "event", "id"])
    df_anomaly = pd.DataFrame(anomaly_labels, columns = ['id', 'type'])
    df_clean = pd.DataFrame(clean_data, columns = ["icao", "timestamp", "event", "id"])
    df = df.sort_values("timestamp")
    df_clean = df_clean.sort_values("timestamp")

    #destination files for data and anomalies
    df.to_csv(file_location + 'data.csv')
    df.to_csv(file_location + 'clean_data.csv')
    df_anomaly.to_csv(file_location + 'data_anomaly_labels.csv')


def generate_data_only_chaining():
    data = []
    #timestamp is the running record of current timestamp
    timestamp = 0
    anomaly_labels = []
    id = 0
    
    clean_number = 60000
    anomaly_number = 10000

    timestamp, id = normal(clean_number, timestamp, id, data, anomaly_labels)
    timestamp, id = chaining(anomaly_number, timestamp, id, data, anomaly_labels)
    timestamp, id = normal(clean_number, timestamp, id, data, anomaly_labels)

    df = pd.DataFrame(data, columns = ["icao", "timestamp", "event", "id"])
    df_anomaly = pd.DataFrame(anomaly_labels, columns = ['id', 'type'])
    df = df.sort_values("timestamp")
    
    #destination files for data and anomalies
    df.to_csv(file_location + 'data_chaining_only.csv')
    df_anomaly.to_csv(file_location + 'data_chaining_only_anomaly_labels.csv')