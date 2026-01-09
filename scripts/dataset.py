import torch
import torchaudio
from torch.utils.data import Dataset
import os

class OwlSoundDataset(Dataset):
    def __init__(self, metadata_df, audio_dir, fold_filter=None, sample_rate=16000, duration=3.0, channels=3):
        self.channels = channels
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.audio_dir = audio_dir

        # filter by folds (train/validation set)
        if fold_filter is not None:
            self.metadata = metadata_df[metadata_df['fold'].isin(fold_filter)].reset_index(drop=True)
        else:
            self.metadata = metadata_df

        # audio transform 
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=128,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        file_path = os.path.join(self.audio_dir, row['segment'])
        label = row['label']

        waveform, sr = torchaudio.load(file_path)

        # resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # pad or truncate to target length
        if waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :self.target_length]

        # convert to Mel spectrogram
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        # normalize and repeat to 3 channels
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_db = mel_db.repeat(self.channels, 1, 1) # [3, 128, time]

        return mel_db, label