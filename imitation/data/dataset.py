"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import os
import h5py
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

from imitation.utils.general_utils import AttrDict
from imitation.utils.tensor_utils import pad_sequence
from imitation.utils.obs_utils import process_obs_dict

import numpy as np
import torch




# adapted from robomimic: https://github.com/ARISE-Initiative/robomimic.git
class SequenceDataset(torch.utils.data.Dataset):
    SPLIT = AttrDict(train= 0.95, val= 0.05)

    def __init__(
        self,
        data_paths,
        obs_keys_to_modality,
        dataset_keys,
        window_size=1,
        action_horizon=1,
        obs_keys_to_normalize={},
        split='train'
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset
        """
        super(SequenceDataset, self).__init__()

        assert isinstance(data_paths, list), "data_paths must be a list of paths"

        self.hdf5_paths = [os.path.expanduser(hdf5_pth) for hdf5_pth in data_paths]
        self.hdf5_use_swmr = True
        self._hdf5_files = None
        self.split = split
        self.hdf5_cache_mode = "low_dim"

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys_to_modality.keys())
        self.obs_keys_to_modality = obs_keys_to_modality
        self.dataset_keys = tuple(dataset_keys)

        self.window_size = window_size
        self.action_horizon = action_horizon

        self.load_demo_info()

        # maybe prepare for observation normalization
        self.normalization_stats = self.compute_normalization_stats(list(obs_keys_to_normalize.keys()))

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode == "low_dim":
            # only store low-dim observations
            obs_keys_in_memory = []
            for k in self.obs_keys:
                if obs_keys_to_modality[k] == "low_dim":
                    obs_keys_in_memory.append(k)

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_files=self.hdf5_files,
                obs_keys=obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
            )

            self.keys_in_memory = set(obs_keys_in_memory).union(set(self.dataset_keys))
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    def load_demo_info(self):
        self.demos = []
        self.n_demos = 0
        
        # keep internal index maps to know which transitions belong to which demos
        self._index_to_file_id = dict()  # maps every index to a file id
        self._index_to_demo_id = []  # maps every index to a demo id
        self._demo_id_to_start_indices = []  # gives start index per demo id
        self._demo_id_to_demo_length = []

        # determine index mapping
        self.total_num_sequences = 0
        
        for file_idx, hdf5_file in enumerate(self.hdf5_files):
            self.demos.append(list(hdf5_file["data"].keys()))

            # sort demo keys
            inds = np.argsort([int(elem[5:]) for elem in self.demos[-1]])
            self.demos[-1] = [self.demos[-1][i] for i in inds]

            self.n_demos += len(self.demos[-1])
            
            self._index_to_demo_id.append({})  # maps every index to a demo id
            self._demo_id_to_start_indices.append({})  # gives start index per demo id
            self._demo_id_to_demo_length.append({})

            for ep in self.demos[-1]:
                demo_length = hdf5_file[f"data/{ep}/actions"][:].shape[0]
                self._demo_id_to_start_indices[-1][ep] = self.total_num_sequences
                self._demo_id_to_demo_length[-1][ep] = demo_length

                num_sequences = demo_length

                for _ in range(num_sequences):
                    self._index_to_demo_id[-1][self.total_num_sequences] = ep
                    self._index_to_file_id[self.total_num_sequences] = file_idx
                    self.total_num_sequences += 1
        
        self.train_split = int(self.SPLIT.train*self.total_num_sequences)
        self.val_split = self.total_num_sequences - self.train_split

        if self.split == 'train':
            self.total_num_sequences = self.train_split
        elif self.split == 'val':
            self.total_num_sequences = self.val_split
        elif self.split == 'all':
            self.total_num_sequences = self.total_num_sequences
        
    @property
    def hdf5_files(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_files is None:
            self._hdf5_files = [h5py.File(hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest') for hdf5_path in self.hdf5_paths]
        return self._hdf5_files

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_files is not None:
            for i in range(len(self._hdf5_files)):
                self._hdf5_files[i].close()
        self._hdf5_files = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_files is None
        yield self.hdf5_files
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_files, obs_keys, dataset_keys):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = []
        for file_idx in range(len(demo_list)):
            print("SequenceDataset: loading dataset into memory...")
            all_data.append({})
            for ep in tqdm(demo_list[file_idx]):
                all_data[file_idx][ep] = {}
                all_data[file_idx][ep]["attrs"] = {}
                all_data[file_idx][ep]["attrs"]["num_samples"] = hdf5_files[file_idx][f"data/{ep}/actions"][:].shape[0]
                # get obs
                all_data[file_idx][ep]["obs"] = {k: hdf5_files[file_idx][f"data/{ep}/obs/{k}"][:] for k in obs_keys}
                # get other dataset keys
                for k in dataset_keys:
                    if k in hdf5_files[file_idx][f"data/{ep}"]:
                        all_data[file_idx][ep][k] = hdf5_files[file_idx][f"data/{ep}/{k}"][:].astype('float32')
                    else:
                        all_data[file_idx][ep][k] = np.zeros((all_data[file_idx][ep]["attrs"]["num_samples"], 1), dtype=np.float32)
        return all_data

    def compute_normalization_stats(self, obs_keys_to_normalize):
        """
        Computes a dataset-wide mean and standard deviation for the observations 
        (per dimension and per obs key) and returns it.
        """
        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = { k : {} for k in traj_obs_dict }
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                if (k in self.obs_keys_to_modality) and (self.obs_keys_to_modality[k] == "rgb"):
                    traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=(0, 2, 3), keepdims=True) # [1, ...]
                    traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=(0, 2, 3), keepdims=True) # [1, ...]
                    traj_stats[k]["min"] = traj_obs_dict[k].min(axis=(0, 2, 3), keepdims=True) # [1, ...]
                    traj_stats[k]["max"] = traj_obs_dict[k].max(axis=(0, 2, 3), keepdims=True) # [1, ...]
                else:
                    traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
                    traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
                    traj_stats[k]["min"] = traj_obs_dict[k].min(axis=0, keepdims=True) # [1, ...]
                    traj_stats[k]["max"] = traj_obs_dict[k].max(axis=0, keepdims=True) # [1, ...]
                    
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                min_v = np.minimum(traj_stats_a[k]["min"], traj_stats_b[k]["min"])
                max_v = np.maximum(traj_stats_a[k]["max"], traj_stats_b[k]["max"])
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2, min=min_v, max=max_v)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        merged_stats = None
        for demo_idx, demo in enumerate(self.demos):
            for ep_idx, ep in enumerate(demo):
                obs_traj = {k: self.hdf5_files[demo_idx][f"data/{ep}/obs/{k}"][()].astype('float32') for k in obs_keys_to_normalize}
                obs_traj = process_obs_dict(obs_traj, self.obs_keys_to_modality)
                
                for dataset_key in self.dataset_keys:
                    obs_traj[dataset_key] = self.hdf5_files[demo_idx][f"data/{ep}/{dataset_key}"][()].astype('float32')

                traj_stats = _compute_traj_stats(obs_traj)
                if merged_stats is None:
                    merged_stats = traj_stats
                else:
                    merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        normalization_stats = {k : {} for k in merged_stats}
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            normalization_stats[k]["mean"] = merged_stats[k]["mean"].astype(np.float32)
            normalization_stats[k]["std"] = (np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-6).astype(np.float32)
            normalization_stats[k]["min"] = merged_stats[k]["min"].astype(np.float32)
            normalization_stats[k]["max"] = merged_stats[k]["max"].astype(np.float32)
        return normalization_stats

    def get_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.

        Returns:
            normalization_stats (dict): a dictionary for normalization.
        """
        return deepcopy(self.normalization_stats)

    def get_dataset_for_ep(self, file_id, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        
        if key in self.keys_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[file_id][ep][key1][key2]
            else:
                ret = self.hdf5_cache[file_id][ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.hdf5_files[file_id][hd5key]
        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        if self.split == 'val':
            index += self.train_split

        # if self.hdf5_cache_mode == "all":
        #     return self.getitem_cache[index]

        return self.get_item(index)

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        file_id = self._index_to_file_id[index]
        demo_id = self._index_to_demo_id[file_id][index]
        demo_start_index = self._demo_id_to_start_indices[file_id][demo_id]

        # start at offset index if not padding for frame stacking
        index_in_demo = index - demo_start_index

        traj = self.get_sequence_from_demo(
            file_id,
            demo_id,
            index_in_demo=index_in_demo
        )

        traj["idx"] = index_in_demo

        return traj

    def get_sequence_from_demo(self, file_id, demo_id, index_in_demo):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
        Returns:
            a dictionary of extracted items.
        """

        demo_length = self._demo_id_to_demo_length[file_id][demo_id]
        assert index_in_demo < demo_length

        traj = {'obs': {}}

        # determine begin and end of sequence
        history_ids = np.arange(index_in_demo - self.window_size, index_in_demo) + 1
        timestep_pad_mask = history_ids >= 0 # True if timestep is valid
        history_ids = np.maximum(history_ids, 0)

        for k in self.obs_keys:
            traj['obs'][k] = self.get_dataset_for_ep(file_id, demo_id, f"obs/{k}")[:]
            traj['obs'][k] = traj['obs'][k][history_ids]

        traj['obs']['timestep_pad_mask'] = timestep_pad_mask
        
        action_chunk_indices = history_ids[:, None] + np.arange(self.action_horizon)
        action_chunk_indices = np.minimum(action_chunk_indices, demo_length - 1)

        traj['actions'] = self.get_dataset_for_ep(file_id, demo_id, "actions")[:]
        traj['actions'] = traj['actions'].take(action_chunk_indices, axis=0)

        return traj


class WeightedMultiDataset(SequenceDataset):
    def __init__(
        self,
        data_paths,  # Expecting exactly 2 paths
        obs_keys_to_modality,
        dataset_keys,
        target_prob=0.5,  # Probability of sampling from the first file
        **kwargs
    ):
        assert len(data_paths) == 2, "This class expects exactly two HDF5 file paths."
        
        super().__init__(
            data_paths=data_paths,
            obs_keys_to_modality=obs_keys_to_modality,
            dataset_keys=dataset_keys,
            **kwargs
        )
        
        self.target_prob = target_prob
        
        # Separate the indices by file_id for sampling
        self.file_0_indices = [idx for idx, f_id in self._index_to_file_id.items() if f_id == 0]
        self.file_1_indices = [idx for idx, f_id in self._index_to_file_id.items() if f_id == 1]
        
        # Adjust for split if necessary
        if self.split == 'train':
            self.file_0_indices = [i for i in self.file_0_indices if i < self.train_split]
            self.file_1_indices = [i for i in self.file_1_indices if i < self.train_split]
        elif self.split == 'val':
            # Note: SequenceDataset shifts index in __getitem__ for 'val', 
            # so we store the raw indices here.
            self.file_0_indices = [i for i in self.file_0_indices if i >= self.train_split]
            self.file_1_indices = [i for i in self.file_1_indices if i >= self.train_split]

    def __getitem__(self, index):
        """
        Overrides the standard indexing. Instead of using the passed 'index',
        it uses the index as a seed/trigger to sample from one of the two files
        based on the defined probability.

        """
        #import pdb; pdb.set_trace()
        # Determine which file to sample from
        if np.random.rand() < self.target_prob:
            # Sample a random index from the first file's available indices
          #  print("Sampling from file 0")
            sampled_idx = np.random.choice(self.file_0_indices)
        else:
            # Sample a random index from the second file
           # print("Sampling from file 1")
            sampled_idx = np.random.choice(self.file_1_indices)

        # Handle the internal 'val' offset logic from the parent class
        if self.split == 'val':
            # The parent get_item expects a relative index for val
            return self.get_item(sampled_idx - self.train_split)
            
        return self.get_item(sampled_idx)