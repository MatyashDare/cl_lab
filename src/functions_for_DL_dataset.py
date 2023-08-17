import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import Dataset, T_co
from torch.utils.data import DataLoader


class AlcDataset(Dataset):
    """
    A custom Dataset implementation for our representations of recordings
    from the ALC corpus
    """

    def __init__(self, df, cols, target, path):
        """
        Builds a torch Dataset containing representations of recordings from
        the ALC corpus

        """
        self.df = df
        self.cols = cols
        self.target = target
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> T_co:
        """
        :return: a recording label, its representation, the length of features, paths

        """
        # target to simple one-hot encodings
        t = self.df.iloc[index][self.target]
        if t == 0:
            t = [1,0]
        else:
            t = [0,1]
        # features to Tensors
        f = torch.Tensor(self.df.iloc[index][self.cols])
        # lengths of audio represenattion vectors
        l = torch.Tensor(self.df.loc[index][self.cols][0]).shape[0]
        # full_paths to audio_files for mapping
        p = self.df.iloc[index][self.path]

        return t, f, l, p

    
def collate_function(batch):
    # open batch with 
    # (target, features, length of features, paths to audios, extra info)
    # -> append in to lists
    targets, features, lengths, paths, extras = [[] for _ in range(5)]
    for (t, f, l, p) in batch:
        targets.append(t)
        features.append(f)
        lengths.append(l)
        paths.append(p)

    targets = torch.tensor(targets)
    # flatten feature list -> pad
    pad_flat_features = pad_sequence([ft for feats in features for ft in feats], batch_first=True)
    # change dimensions of matrices for LSTM input
    final_features = torch.permute(torch.stack(torch.split(pad_flat_features, len(features[0]))),
                                   (0, 2, 1))
    lengths = torch.LongTensor(lengths)

    return targets, final_features, lengths, paths


def strings_to_vectors(df, feature_cols, feature_length=10**4):
    for col in feature_cols:
        df[col] = df[col].apply(lambda x: [float(k) for k in x[1:-1].split(', ')][:feature_length])
    return df