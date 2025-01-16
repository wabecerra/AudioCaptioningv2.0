import os
import sys
import torch
import torchaudio
import numpy as np
import nltk
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from transformers import RobertaTokenizer
from tqdm import tqdm
from niacin.augment import RandAugment
from niacin.text import en
import matplotlib.pyplot as plt

# NOTE: If you need the 'omw-1.4' resource, consider placing this in main.py or environment setup:
# nltk.download('omw-1.4')

def plot_spectrogram(spec: torch.Tensor) -> None:
    """
    Plot a spectrogram using matplotlib.
    
    Args:
        spec (torch.Tensor): The spectrogram to visualize.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Spectrogram')
    plt.show()


def augment_spectrogram(spec: np.ndarray) -> torch.Tensor:
    """
    Apply a series of augmentations to the input spectrogram.
    
    Args:
        spec (np.ndarray): The raw spectrogram data (shape: time x freq).
    
    Returns:
        torch.Tensor: Augmented spectrogram as a PyTorch tensor.
    """
    spec_tensor = torch.tensor(spec).T.unsqueeze(0)  # (1, freq, time)
    time_stretch = torchaudio.transforms.TimeStretch(n_freq=64, fixed_rate=1.8)
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)
    
    augmented = torch.abs(time_stretch(spec_tensor))
    augmented = freq_mask(augmented)
    augmented = time_mask(augmented)
    return augmented


def augment_caption(caption: str) -> str:
    """
    Apply textual augmentations to a given caption.
    
    Args:
        caption (str): The raw text caption.
    
    Returns:
        str: Augmented caption string.
    """
    augmentor = RandAugment([
        en.add_synonyms,
        en.add_hypernyms,
        en.remove_articles,
        en.remove_contractions,
        en.remove_punctuation
    ], n=2, m=10, shuffle=False)
    
    for transform_fn in augmentor:
        caption = transform_fn(caption)
    return caption


def load_data_from_npy(
    data_dir: str,
    split: str
) -> dict:
    """
    Load spectrograms and captions from .npy files in the specified directory.
    The data is split into train/val or test sets.
    
    Args:
        data_dir (str): Path to the dataset directory.
        split (str): One of ['train/val', 'test'] indicating which set to load.
    
    Returns:
        dict: A dictionary with keys:
            {
                'train_spectrograms', 'train_captions',
                'val_spectrograms', 'val_captions',
                'test_spectrograms', 'test_captions'
            }
    """
    spectrograms = []
    captions = []
    train_spectrograms = []
    train_captions = []
    val_spectrograms = []
    val_captions = []
    test_spectrograms = []
    test_captions = []

    if split == 'train/val':
        data_dir = os.path.join(data_dir, 'clotho_dataset_dev')
    elif split == 'test':
        data_dir = os.path.join(data_dir, 'clotho_dataset_eva')
    else:
        raise ValueError("split should be 'train/val' or 'test'.")

    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".npy"):
            item = np.load(os.path.join(data_dir, file), allow_pickle=True)
            spectrograms.append(item.features[0])
            # Clean up caption by removing <s> and </s>
            caption = item.caption[0][6:-6]
            captions.append(caption)

    if split == 'train/val':
        n = len(spectrograms)
        train_spectrograms = spectrograms[: int(n * 0.8)]
        val_spectrograms = spectrograms[int(n * 0.8) :]
        train_captions = captions[: int(n * 0.8)]
        val_captions = captions[int(n * 0.8) :]
    else:  # split == 'test'
        test_spectrograms = spectrograms
        test_captions = captions

    return {
        'train_spectrograms': train_spectrograms,
        'train_captions': train_captions,
        'val_spectrograms': val_spectrograms,
        'val_captions': val_captions,
        'test_spectrograms': test_spectrograms,
        'test_captions': test_captions
    }


class AudioCaptioningDataset(Dataset):
    """
    Custom Dataset for audio-caption pairs. Each item is a spectrogram 
    (with optional augmentation) and its corresponding text caption.

    Args:
        spectrograms (List[np.ndarray]): List of raw spectrogram data.
        captions (List[str]): List of text captions.
        augment (bool): Whether to apply data augmentation to spectrograms/captions.
        multichannel (bool): If True, replicate the spectrogram channel thrice for models 
                             expecting 3-channel input. 
    """
    def __init__(
        self,
        spectrograms: list,
        captions: list,
        augment: bool = False,
        multichannel: bool = False
    ):
        # From config file or known constraints:
        self.spectrogram_shape = (64, (40 * 44100 + 1) // 512)
        self.caption_shape = (30, 1)

        self.augment = augment
        self.spectrograms = spectrograms
        self.captions = captions
        self.multichannel = multichannel
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int):
        spec = self.spectrograms[idx]
        caption = self.captions[idx]

        # Augment or simply transform
        if self.augment:
            spec = augment_spectrogram(spec)
            caption = augment_caption(caption)
        else:
            spec = torch.tensor(spec).T.unsqueeze(0)  # (1, freq, time)

        # Pad the spectrogram to a fixed shape
        spec_pad = torch.zeros(self.spectrogram_shape)
        spec_pad[: spec.size(1), : spec.size(2)] = spec
        spec = spec_pad

        # Tokenize the caption
        tokenizer_output = self.tokenizer(
            caption,
            return_tensors='pt',
            padding='max_length',
            max_length=self.caption_shape[0],
            truncation=True
        )
        input_ids = tokenizer_output['input_ids']
        attention_mask = tokenizer_output['attention_mask']

        # If model expects 3-channel data
        if self.multichannel:
            spec = torch.stack([spec, spec, spec], dim=0)

        return spec, input_ids, attention_mask


if __name__ == "__main__":
    # Example usage / test
    data_dir = '/content/drive/My Drive/MusicCaptioning/dataset/'
    data_train = load_data_from_npy(data_dir, 'train/val')
    data_test = load_data_from_npy(data_dir, 'test')

    train_dataset = AudioCaptioningDataset(
        data_train['train_spectrograms'], data_train['train_captions'], augment=True
    )
    val_dataset = AudioCaptioningDataset(
        data_train['val_spectrograms'], data_train['val_captions']
    )
    test_dataset = AudioCaptioningDataset(
        data_test['test_spectrograms'], data_test['test_captions']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for batch_idx, (spec, input_ids, attention_mask) in enumerate(train_dataloader):
        print(spec.shape, input_ids.shape, attention_mask.shape)
        if batch_idx == 1:
            break
