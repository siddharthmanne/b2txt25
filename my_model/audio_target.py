import torch
from torch import nn
import librosa
import numpy as np
from transformers import pipeline, Qwen2AudioForConditionalGeneration, AutoProcessor

class AudioTarget(nn.Module):
    '''
    Frozen target pipeline: Text → TTS → Audio → AudioTower → audio_embedding

    This creates the target embeddings that brain_embedding should align to.
    All components are FROZEN (no training).
    '''
    def __init__(
        self,
        t2a_model_id="facebook/mms-tts-eng",
        shared_a2t_model=None,
        shared_processor=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        '''
        t2a_model_id (str)       - Text-to-Audio (TTS) model ID
        shared_a2t_model         - Shared Qwen2AudioForConditionalGeneration instance
        shared_processor         - Shared AutoProcessor instance
        device (str)             - Device to run on
        '''
        super(AudioTarget, self).__init__()

        self.device = device
        self.t2a_model_id = t2a_model_id

        # Text-to-Speech pipeline (for converting labels to audio)
        self.t2a_pipeline = pipeline("text-to-speech", t2a_model_id, device=device)

        # Use shared model and processor (passed from Brain2TextModel)
        if shared_a2t_model is None or shared_processor is None:
            raise ValueError("AudioTarget requires shared_a2t_model and shared_processor to be provided")

        self.a2t_model = shared_a2t_model
        self.audio_processor = shared_processor

        # Extract audio tower component
        self.audio_tower = self.a2t_model.audio_tower  # This creates audio embeddings

        # Get target sampling rate
        self.target_sr = self.audio_processor.feature_extractor.sampling_rate

        # Freeze all parameters in audio_tower
        for param in self.audio_tower.parameters():
            param.requires_grad = False

        self.audio_tower.eval()  # Set to eval mode

    def text_to_audio_embedding(self, text):
        '''
        Convert text to audio array using TTS and audio array to audio embedding using AudioTower

        Args:
            text (str) - Text transcription

        Returns:
            audio_embedding (torch.Tensor) - [1, seq_len, 1280] where seq_len = num_frames // 2
        '''
        # Convert raw text to audio array
        with torch.no_grad():
            tts_output = self.t2a_pipeline(text)  # dict with 'audio' and 'sampling_rate'

        # Extract audio array and sampling rate
        audio_array = tts_output['audio']  # shape: [1, n_samples]
        source_sr = tts_output['sampling_rate']  # scalar int

        # TTS outputs 2D array (1, n_samples) - squeeze to 1D
        audio_array = audio_array.squeeze()  # shape: [n_samples]

        # Resample if needed
        if source_sr != self.target_sr:
            audio_array = audio_array.astype(np.float32)  # shape: [n_samples]
            audio_array = librosa.resample(
                audio_array,
                orig_sr=source_sr,
                target_sr=self.target_sr
            )  # shape: [n_samples_resampled]

        # Convert audio array to audio embedding

        # Process audio with Qwen2Audio processor - expects 16kHz audio
        # Processor converts to mel-spectrogram: 128 mel-frequency bins
        inputs = self.audio_processor(
            text=["<|AUDIO|>"],  # Required token for audio input, list with 1 element
            audios=[audio_array],  # List with 1 audio array: [n_samples] at 16kHz
            sampling_rate=self.target_sr,  # Must be 16000
            return_tensors="pt"
        ).to(self.device)
        # inputs.input_features: [1, 128, num_frames] - mel-spectrogram with 128 mel bins

        # Encode audio to embeddings using audio_tower (Whisper-large-v3 encoder)
        # FROZEN, no gradients
        with torch.no_grad():
            features = inputs.input_features.to(dtype=self.a2t_model.dtype)  # [1, 128, num_frames]
            encoder_outputs = self.audio_tower(
                input_features=features,
                return_dict=True
            )
            # Encoder has stride-2 pooling layer, so output seq_len = num_frames // 2
            # Each output frame corresponds to ~40ms of original audio
            audio_embedding = encoder_outputs.last_hidden_state  # [1, seq_len, 1280]

        return audio_embedding  # [1, seq_len, 1280] - audio embeddings in Whisper encoder space

    def forward(self, text_labels):
        '''
        End-to-end: Text labels → Audio → Audio embeddings

        Args:
            text_labels (list of str) - Batch of text transcriptions, length = batch_size

        Returns:
            audio_embeddings (torch.Tensor) - [batch, max_seq_len, 1280]
            attention_mask (torch.Tensor) - [batch, max_seq_len]
        '''
        batch_embeddings = []  # Will store list of [seq_len_i, 1280] tensors

        for text in text_labels:  # Process each text individually
            audio_embedding = self.text_to_audio_embedding(text)  # [1, seq_len_i, 1280]
            batch_embeddings.append(audio_embedding.squeeze(0))  # [seq_len_i, 1280]

        # Pad to max sequence length for embeddings
        max_seq_len = max(emb.size(0) for emb in batch_embeddings)  # scalar int
        embedding_dim = batch_embeddings[0].size(-1)  # 1280
        batch_size = len(batch_embeddings)  # scalar int

        audio_embeddings = torch.zeros(
            batch_size, max_seq_len, embedding_dim,
            dtype=batch_embeddings[0].dtype,
            device=batch_embeddings[0].device
        )  # [batch, max_seq_len, 1280]

        attention_mask = torch.zeros(
            batch_size, max_seq_len,
            dtype=torch.long,
            device=batch_embeddings[0].device
        )  # [batch, max_seq_len]

        for i, emb in enumerate(batch_embeddings):  # emb: [seq_len_i, 1280]
            seq_len = emb.size(0)  # scalar int
            audio_embeddings[i, :seq_len, :] = emb  # Fill in [seq_len_i, 1280]
            attention_mask[i, :seq_len] = 1  # Mark valid positions


        return audio_embeddings, attention_mask  # [batch, max_seq_len, 1280], [batch, max_seq_len]
