import h5py
import numpy as np

fist_file = './Data/fist_gesture_features.h5'
circle_file = './Data/finger_circle_features.h5'

output_file = './Data/combined_gesture_features.h5'

with h5py.File(fist_file, 'r') as f1:
    fist_features = f1['fist_gesture_features'][:]
    fist_columns = ['Gesture', 'Fist_State']

with h5py.File(circle_file, 'r') as f2:
    circle_features = f2['features'][:]
    circle_columns = ['Gesture', 'Float_Value', 'Angle_Diff']

combined_features = np.concatenate(
    [fist_features, circle_features],
    axis=1
)

combined_columns = fist_columns + circle_columns[1:]

with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('combined_features', data=combined_features, compression='gzip')
    hf.create_dataset('columns', data=np.array(combined_columns, dtype='S'))

print(f'Combined dataset saved to {output_file}')
