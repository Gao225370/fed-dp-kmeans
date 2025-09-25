# In merge_hdf5.py (Corrected Version)

import os
import h5py
import argparse
from collections import defaultdict


def merge_files(data_dir):
    """
    Merges HDF5 files in a directory that have a '_pid*.hdf5' suffix
    into a single file without the suffix.
    """
    files_to_merge = defaultdict(list)
    for filename in os.listdir(data_dir):
        if '_pid' in filename and filename.endswith('.hdf5'):
            # Corrected base name logic
            base_name = filename.split('_pid')[0] + '.hdf5'
            files_to_merge[base_name].append(filename)

    if not files_to_merge:
        print(f"No PID-suffixed files found to merge in {data_dir}")
        return

    for target_filename, source_filenames in files_to_merge.items():
        target_path = os.path.join(data_dir, target_filename)
        print(f"Creating merged file: {target_path}")

        with h5py.File(target_path, 'w') as f_target:
            for source_filename in source_filenames:
                source_path = os.path.join(data_dir, source_filename)
                print(f"  > Merging from: {source_filename}")
                with h5py.File(source_path, 'r') as f_source:
                    # Iterate through groups in source (e.g., 'train', 'val')
                    for group_name in f_source:
                        # Iterate through users in the group
                        for user_id in f_source[group_name]:
                            source_group_path = f'/{group_name}/{user_id}'

                            # --- 关键修复：使用 f_source.copy 而不是 h5py.copy ---
                            # Also, ensure the target group exists before copying into it
                            if group_name not in f_target:
                                f_target.create_group(group_name)
                            f_source.copy(source_group_path, f_target[group_name], name=user_id)
                            # --- 修复结束 ---

        print(f"Finished merging {len(source_filenames)} files into {target_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PID-suffixed HDF5 files.")
    parser.add_argument('data_dir', type=str, help='Directory containing the HDF5 files to merge.')
    args = parser.parse_args()
    merge_files(args.data_dir)