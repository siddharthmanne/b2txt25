import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import h5py
import hashlib
import os
from typing import Optional

class AudioTarget(nn.Module):
    '''
    Frozen target pipeline: Text → Precomputed Audio Embeddings

    This loads precomputed audio embeddings for text labels.
    All embeddings must be precomputed using precompute_all_embeddings.py before training.

    Benefits:
    - Extremely fast (no TTS or AudioTower computation during training)
    - Deterministic (same text always gives same embedding)
    - Memory efficient (embeddings loaded on-demand from HDF5)
    '''
    def __init__(
        self,
        cache_dir: str,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=None,
    ):
        '''
        cache_dir (str)   - Directory containing embeddings.h5 file
        device (str)      - Device to load embeddings to
        logger            - Logger instance
        '''
        super(AudioTarget, self).__init__()

        self.device = device
        self.logger = logger

        # Setup embeddings file path
        if cache_dir is None:
            raise ValueError("cache_dir must be provided")

        self.embeddings_path = os.path.join(cache_dir, 'embeddings.h5')

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found at {self.embeddings_path}\n"
                f"Please run: python precompute_all_embeddings.py"
            )

        # Log cache info
        if logger:
            with h5py.File(self.embeddings_path, 'r') as f:
                n_entries = len(f['embeddings'].keys())
            logger.info(f"Audio embedding cache loaded from: {self.embeddings_path}")
            logger.info(f"  Cached embeddings: {n_entries}")

    @staticmethod
    def _text_to_hash(text):
        """Convert text to SHA256 hash for cache lookup."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()  

    def _load_embedding_from_cache(self, text):
        '''
        Load precomputed embedding from HDF5 cache.

        Args:
            text (str) - Text transcription

        Returns:
            embedding (torch.Tensor) - [1, seq_len, 1280]
            attention_mask (torch.Tensor) - [1, seq_len]
        '''
        text_hash = self._text_to_hash(text)

        with h5py.File(self.embeddings_path, 'r') as f:
            if text_hash not in f['embeddings']:
                raise KeyError(
                    f"Text not found in cache: '{text[:50]}...'\n"
                    f"Please rerun precompute_all_embeddings.py to include this text."
                )

            # Load embedding and mask
            embedding_data = f['embeddings'][text_hash]['embedding'][:]
            attention_mask_data = f['embeddings'][text_hash]['attention_mask'][:]

        # Convert to torch tensors and add batch dimension
        embedding = torch.from_numpy(embedding_data).unsqueeze(0)  # [1, seq_len, 1280]
        attention_mask = torch.from_numpy(attention_mask_data).unsqueeze(0)  # [1, seq_len]

        return embedding, attention_mask

    def forward(self, text_labels):
        '''
        Load precomputed audio embeddings for batch of text labels.

        Args:
            text_labels (list of str) - Batch of text transcriptions, length = batch_size

        Returns:
            audio_embeddings (torch.Tensor) - [batch, max_seq_len, 1280]
            attention_mask (torch.Tensor) - [batch, max_seq_len]
        '''
        batch_size = len(text_labels)

        # Load embeddings from cache for each text
        # NOTE: HDF5 file reading MUST be sequential - h5py doesn't support parallel reads
        embeddings_list = []
        masks_list = []

        for text in text_labels:
            embedding, mask = self._load_embedding_from_cache(text)
            # embedding: [1, seq_len, 1280]
            # mask: [1, seq_len]
            embeddings_list.append(embedding.squeeze(0))  # [seq_len, 1280]
            masks_list.append(mask.squeeze(0))  # [seq_len]

        # OPTIMIZED: Use torch's optimized pad_sequence instead of manual padding
        # pad_sequence expects list of [seq_len, dim] tensors
        # Returns [batch, max_seq_len, 1280] with batch_first=True
        audio_embeddings = pad_sequence(
            embeddings_list,
            batch_first=True,
            padding_value=0.0
        )  # [batch, max_seq_len, 1280]

        # For attention mask, pad with 0 (masked positions)
        attention_mask = pad_sequence(
            masks_list,
            batch_first=True,
            padding_value=0
        )  # [batch, max_seq_len]

        # Move to device (single transfer for entire batch - efficient)
        audio_embeddings = audio_embeddings.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        return audio_embeddings, attention_mask  # [batch, max_seq_len, 1280], [batch, max_seq_len]
