import h5py
import numpy as np
import os
from collections import defaultdict
import glob
import re

class Recorder:
    def __init__(self, save_dir="./dataset"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.low_dim_data = defaultdict(list)
        self.image_data = defaultdict(lambda: defaultdict(list))  # cam_name -> {"images": [...], "timestamp_ns": [...]}

    def clear(self):
        self.low_dim_data.clear()
        self.image_data.clear()

    def add_low_dim_step(self, state_dict):
        for key, value in state_dict.items():
            self.low_dim_data[key].append(value)

    def add_image_step(self, cam_name, img, timestamp_ns):
        self.image_data[cam_name]["images"].append(img)
        self.image_data[cam_name]["timestamp_ns"].append(timestamp_ns)

    def _save_recursive(self, h5_group, key, data_list):
        """Recursively saves data, creating groups for nested dictionaries."""
        # Check if the first item in the list is a dictionary
        if isinstance(data_list[0], dict):
            sub_group = h5_group.create_group(key)
            # Find all keys present in the sub-dictionaries
            sub_keys = data_list[0].keys()
            for sk in sub_keys:
                # Extract the list of values for this specific sub-key
                sub_data_list = [d[sk] for d in data_list]
                self._save_recursive(sub_group, sk, sub_data_list)
        else:
            # Base case: it's a list of arrays/numbers, so save as dataset
            data = np.array(data_list)
            print(f"SAVING episode key: {key} of size {data.shape}")
            h5_group.create_dataset(key, data=data, compression="gzip")

    def get_next_episode_index(self):
        pattern = os.path.join(self.save_dir, "episode_*.hdf5")
        existing = glob.glob(pattern)

        max_idx = -1
        regex = re.compile(r"_(\d+)\.hdf5$")

        for filepath in existing:
            filename = os.path.basename(filepath)
            match = regex.search(filename)
            if match:
                idx = int(match.group(1))
                max_idx = max(max_idx, idx)

        return max_idx + 1

    def save_episode(self, episode_name):
        if not self.low_dim_data or not self.image_data:
            print("one or both of low_dim / image is missing, not saving.")
            return

        filepath = os.path.join(self.save_dir, f"{episode_name}.hdf5")
        with h5py.File(filepath, "w") as f:
            if self.low_dim_data:
                low_dim_group = f.create_group("low_dim")
                for key, data_list in self.low_dim_data.items():
                    self._save_recursive(low_dim_group, key, data_list)

            if self.image_data:
                images_group = f.create_group("images")
                for cam_name, cam_dict in self.image_data.items():
                    cam_group = images_group.create_group(cam_name)
                    imgs = np.array(cam_dict["images"])
                    print(f"SAVING episode key: images/{cam_name}/images of size {imgs.shape}")
                    cam_group.create_dataset("images", data=imgs, compression="gzip")

                    ts_name = cam_name.replace("_image", "_img") + "_timestamp_ns"
                    ts_data = np.array(cam_dict["timestamp_ns"], dtype=np.int64)
                    print(f"SAVING episode key: images/{cam_name}/{ts_name} of size {ts_data.shape}")
                    cam_group.create_dataset(ts_name, data=ts_data, compression="gzip")

        print(f"[RECORDER] Saved {episode_name} to {filepath}")
        self.clear()

    def load_episode(self, episode_idx):
        """Loads episode_{episode_idx}.hdf5 and returns the trajectory buffer. """
        if episode_idx is None:
            return []

        filepath = os.path.join(self.save_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Episode file not found: {filepath}")
        with h5py.File(filepath, "r") as f:
            follower_jp = f["low_dim"]["follower_jp"][:]          # shape (n_steps, 7)
            gripper_pos = f["low_dim"]["gripper_pos"][:]           # shape (n_steps,)
            data = np.concatenate([follower_jp, gripper_pos[:, None]], axis=1)
        trajectory_buffer = [np.array(step) for step in data]
        return trajectory_buffer
