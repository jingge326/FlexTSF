import numpy as np
import pickle
from scipy.ndimage import zoom
import os
import csv
from pathlib import Path


def concat(sample, concat_keys):
    inputs = np.concatenate([sample[key] for key in concat_keys], axis=-1)
    sample["inputs"] = inputs
    sample = {key: sample[key]
              for key in sample.keys() if key not in concat_keys}
    return sample


def rescale(image, new_h, new_w):
    # Rescale the image to the new size using bilinear interpolation
    # image shape: (batch, height, width, channels)
    batch, height, width, channels = image.shape
    zoom_factors = (1, new_h / height, new_w / width, 1)
    # order=1 for bilinear interpolation
    img_resized = zoom(image, zoom_factors, order=1)
    return img_resized


def to_array(sample):
    if 'B01' in sample.keys():
        x10 = np.stack([sample[key].astype(np.float32)
                       for key in ['B04', 'B03', 'B02', 'B08']], axis=-1)
        x20 = np.stack([sample[key].astype(np.float32) for key in [
                       'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']], axis=-1)
        x60 = np.stack([sample[key].astype(np.float32)
                       for key in ['B01', 'B09', 'B10']], axis=-1)
        doy = np.array(sample['doy']).astype(np.float32)
        labels = sample['labels'].astype(
            np.float32)[np.newaxis, ..., np.newaxis]
        sample = {"x10": x10, "x20": x20,
                  "x60": x60, "day": doy, "labels": labels}
        return sample
    # else:
    sample['x10'] = sample['x10'].astype(np.float32)
    sample['x20'] = sample['x20'].astype(np.float32)
    sample['x60'] = sample['x60'].astype(np.float32)
    sample['day'] = sample['day'].astype(np.float32)
    sample['labels'] = sample['labels'].astype(np.int64)[..., np.newaxis]
    return sample


def construct_time_series(file_path, img_size):
    with open(file_path, 'rb') as handle:
        sample = pickle.load(handle, encoding='latin1')

    sample = to_array(sample)
    # sample['x10'] = sample['x10'] * 1e-4
    # sample['x20'] = sample['x20'] * 1e-4
    # sample['x60'] = sample['x60'] * 1e-4

    sample['x10'] = sample['x10']
    sample['x20'] = sample['x20']
    sample['x60'] = sample['x60']

    sample['x10'] = rescale(sample['x10'], img_size, img_size)
    sample['x20'] = rescale(sample['x20'], img_size, img_size)
    sample['x60'] = rescale(sample['x60'], img_size, img_size)

    sample = concat(sample, ['x10', 'x20', 'x60'])

    time = sample['day']
    data = sample['inputs'].squeeze()

    return time, data


def save_arrays(time_array, data_array, mask_array, output_dir, set_name):
    # Save the arrays to local storage
    np.save(os.path.join(output_dir, f'{set_name}_time.npy'), time_array)
    np.save(os.path.join(output_dir, f'{set_name}_data.npy'), data_array)
    np.save(os.path.join(output_dir, f'{set_name}_mask.npy'), mask_array)
    print(f"Arrays for {set_name} saved to {output_dir}")


def build_dataset(csv_path, folder_path, img_size, output_dir, set_name):
    all_time_raw = []
    all_data_raw = []
    list_time_length = []  # Collect lengths of time arrays

    cnt = 0
    # Read and collect raw data
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            file_name = row[0].strip()
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                print(f"File {file_name} not found.")
                continue

            # Process file
            time, data = construct_time_series(file_path, img_size)
            all_time_raw.append(time)
            all_data_raw.append(data)
            list_time_length.append(len(time))  # Collect time lengths
            cnt += 1
            if cnt % 100 == 0:
                print(f"Constructed {cnt} time series")

    # Calculate maximum time length
    max_time_len = max(list_time_length)

    # Pad time and data arrays, and create masks
    padded_time = []
    padded_data = []
    masks = []

    cnt = 0
    for time, data in zip(all_time_raw, all_data_raw):
        # Pad time array
        padded_time.append(
            np.pad(time, (0, max_time_len - len(time)), mode='constant', constant_values=0))

        # Pad data array along the time dimension (assuming time is the first dimension)
        pad_width = [(0, max_time_len - data.shape[0])] + \
            [(0, 0)] * (len(data.shape) - 1)
        padded_data.append(
            np.pad(data, pad_width, mode='constant', constant_values=0))

        # Create mask (1s for original data, 0s for padding)
        mask = np.zeros_like(padded_data[-1], dtype=np.uint8)
        slices = tuple(slice(0, dim) for dim in data.shape)
        mask[slices] = 1
        masks.append(mask)
        cnt += 1
        if cnt % 100 == 0:
            print(f"Padded {cnt} time series")

    # Stack padded arrays
    all_time = np.stack(padded_time, axis=0)
    all_data = np.stack(padded_data, axis=0)
    all_mask = np.stack(masks, axis=0)

    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_arrays(all_time, all_data, all_mask, output_dir, set_name)

    print(f"Finished processing {set_name} set")


if __name__ == "__main__":
    proj_path = Path(__file__).resolve().parents[2]
    path_raw = proj_path / "data/satsm/raw"
    output_dir = proj_path / "data/satsm/processed"
    img_size = 1

    # Process training, validation, and testing sets
    build_dataset(os.path.join(path_raw, 'train_paths.csv'),
                  path_raw + "/common", img_size, output_dir, 'train')
    build_dataset(os.path.join(path_raw, 'val_paths.csv'),
                  path_raw + "/common", img_size, output_dir, 'val')
    build_dataset(os.path.join(path_raw, 'test_paths.csv'),
                  path_raw + "/common", img_size, output_dir, 'test')
