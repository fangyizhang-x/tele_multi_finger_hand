import numpy as np

# create some sample data
data = np.array([1,2,5,6])

# normalize the data
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

# print the normalized data
print(normalized_data)
