import pickle

# Paths to your Pickle files
circle_pickle_file = 'path/to/circle_data.pkl'
onoff_pickle_file = 'path/to/onoff_data.pkl'
combined_pickle_file = 'path/to/combined_data.pkl'

# Load circle data from Pickle file
with open(circle_pickle_file, 'rb') as f:
    circle_data = pickle.load(f)

# Load onoff data from Pickle file
with open(onoff_pickle_file, 'rb') as f:
    onoff_data = pickle.load(f)

# Combine data
combined_data = {
    'circle': circle_data,
    'onoff': onoff_data
}

# Save combined data to a new Pickle file
with open(combined_pickle_file, 'wb') as f:
    pickle.dump(combined_data, f)

print(f"Combined Pickle data saved to {combined_pickle_file}")

