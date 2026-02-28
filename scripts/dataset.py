import torch
import torchaudio
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            n_fft=512,
            hop_length=256,
            f_min=0,
            f_max=8000
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

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

class OwlSoundWaveformDataset(Dataset):
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

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        file_path = os.path.join(self.audio_dir, row['segment'])
        label = row['label']

        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(DEVICE)

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
        return waveform, label
    
def compute_mel_spec(wav_path, sample_rate=16000, duration=3.0, channels=1):
    mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,
            n_fft=512,
            hop_length=256,
            f_min=0,
            f_max=8000
        )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    waveform, sr = torchaudio.load(wav_path)
    waveform_unmodified = waveform
    target_length = int(sample_rate * duration)

    # resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # pad or truncate to target length
    if waveform.shape[1] < target_length:
        pad_amount = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :target_length]
    
    # convert to Mel spectrogram
    mel = mel_transform(waveform)
    mel_db = db_transform(mel)

    # normalize and repeat to 3 channels
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_db = mel_db.repeat(channels, 1, 1) # [3, 128, time]

    return mel_db, waveform_unmodified, waveform

if __name__ == '__main__':
    mel_label_5, wav_5, wav_proc_5 = compute_mel_spec('../buowset1.3/buowset1.3/0a0f33fc-a72a-4395-9f5e-0fe96b1ff4ae.wav')
    mel_label_3, wav_3, wav_proc_3 = compute_mel_spec('../buowset1.3/buowset1.3/0a1e8cf0-be5b-464a-abb4-0813010a34ff.wav')

    print(wav_proc_5.shape)
    print(wav_proc_3.shape)
    print()
    plt.figure(figsize=(10, 4))
    # Use the first channel for display
    plt.imshow(mel_label_5[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel Frequency Bin")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(10, 4))
    # # Use the first channel for display
    # plt.imshow(mel_label_3[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
    # plt.title("Mel Spectrogram")
    # plt.xlabel("Time (frames)")
    # plt.ylabel("Mel Frequency Bin")
    # plt.colorbar(label="Amplitude (dB)")
    # plt.tight_layout()
    # plt.show()

    print(mel_label_5.shape)
    print(mel_label_3.shape)

    mel_label_5_bytes = mel_label_5.numpy().tobytes()
    mel_label_3_bytes = mel_label_3.numpy().tobytes()

    file_path_1 = "mel_class_5.bin"
    file_path_2 = "mel_class_3.bin"

    try:
        with open(file_path_1, "wb") as file:
            # 3. Write the byte array to the file
            file.write(mel_label_5_bytes)
        print(f"Successfully wrote {len(mel_label_5_bytes)} bytes to {file_path_1}")
    except IOError as e:
        print(f"Error writing to file: {e}")
    
    try:
        with open(file_path_2, "wb") as file:
            # 3. Write the byte array to the file
            file.write(mel_label_3_bytes)
        print(f"Successfully wrote {len(mel_label_3_bytes)} bytes to {file_path_2}")
    except IOError as e:
        print(f"Error writing to file: {e}")
    
    wav_5_bytes = wav_5.numpy().tobytes()
    wav_3_bytes = wav_3.numpy().tobytes()

    file_path_wav_1 = "wav_class_5.bin"
    file_path_wav_2 = "wav_class_3.bin"

    try:
        with open(file_path_wav_1, "wb") as file:
            # 3. Write the byte array to the file
            file.write(wav_5_bytes)
    except IOError as e:
        print(f"Error writing to file: {e}")
    
    try:
        with open(file_path_wav_2, "wb") as file:
            # 3. Write the byte array to the file
            file.write(wav_3_bytes)
    except IOError as e:
        print(f"Error writing to file: {e}")

    wav_proc_5_bytes = wav_proc_5.numpy().tobytes()
    wav_proc_3_bytes = wav_proc_3.numpy().tobytes()


    file_path_wav_1 = "wav_proc_class_5.bin"
    file_path_wav_2 = "wav_proc_class_3.bin"

    try:
        with open(file_path_wav_1, "wb") as file:
            # 3. Write the byte array to the file
            file.write(wav_proc_5_bytes)
    except IOError as e:
        print(f"Error writing to file: {e}")
    
    try:
        with open(file_path_wav_2, "wb") as file:
            # 3. Write the byte array to the file
            file.write(wav_proc_3_bytes)
    except IOError as e:
        print(f"Error writing to file: {e}")

