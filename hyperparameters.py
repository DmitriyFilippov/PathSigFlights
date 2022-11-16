import numpy as np

day = 86400


class hyperparameters:
    normalising_data = False 
    scale_data = True
    preprocess_data_columnwise = False
    normalising_signature = False
    scale_signature = False
    preprocess_signature_columnwise = False
    covariance_method = "raw"
    knn_metric = 'minkowski'
    observed = [np.int_(0), np.int_(1), np.int_(2), np.int_(3), np.int_(4)]
    depth = 2
    time_variable = True
    daytime_variable = True
    stream_length = 7 * day
    stream_spacing = 2 * day

    extrapollation = False
    lead_lag = False
    
    directional_edges = True

    neighbours = 5
 