import h5py

# Paths to your HDF5 files
onoff_file = './Data/fist_gesture_features.h5'
circle_file = './Data/finger_circle_features.h5'

def inspect_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Datasets in {file_path}:")
            for group_name in f:
                print(f"Group: {group_name}")
                group = f[group_name]
                for dataset_name in group:
                    dataset = group[dataset_name]
                    print(f"Dataset: {dataset_name}, Shape: {dataset.shape}, dtype: {dataset.dtype}")

                    # Check if dataset is a chunked dataset and print chunk shape
                    if 'chunks' in dataset.attrs:
                        print(f"Chunk shape: {dataset.attrs['chunks']}")
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")

inspect_hdf5(onoff_file)
inspect_hdf5(circle_file)
