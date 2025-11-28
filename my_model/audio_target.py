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
        a2t_model_id="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        '''
        t2a_model_id (str) - Text-to-Audio (TTS) model ID
        a2t_model_id (str) - Audio-to-Text model ID (we only use AudioTower)
        device (str)       - Device to run on
        '''
        super(AudioTarget, self).__init__()

        self.device = device
        self.t2a_model_id = t2a_model_id
        self.a2t_model_id = a2t_model_id

        # Text-to-Speech pipeline (for converting labels to audio)
        self.t2a_pipeline = pipeline("text-to-speech", t2a_model_id, device=device)

        # Load Qwen Audio model (we only need the audio tower)
        self.a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            a2t_model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # Extract components
        self.audio_processor = AutoProcessor.from_pretrained(a2t_model_id)
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
            audio_embedding (torch.Tensor) - [batch, seq_len, audio_embedding_dim]
        '''
        # Convert raw text to audio array
        with torch.no_grad():
            audio_array = self.t2a_pipeline(text)
        
        # Correct sampling rate
        if audio_array['sampling_rate'] != self.target_sr:
            audio_array = audio_array.astype(np.float32)
            audio_array = librosa.resample(
                audio_array,
                orig_sr=audio_array['sampling_rate'],
                target_sr=self.target_sr
            )
        audio_array = audio_array['audio']

        # Convert audio array to audio embedding

        # Process audio
        inputs = self.audio_processor(
            text=["<|audio_bos|><|audio_eos|>"],  # Dummy text tokens
            audios=[audio_array],
            sampling_rate=self.target_sr,
            return_tensors="pt"
        ).to(self.device)

        # Encode audio to embeddings (FROZEN, no gradients)
        with torch.no_grad():
            features = inputs.input_features.to(dtype=self.a2t_model.dtype)
            encoder_outputs = self.audio_tower(features)
            audio_embedding = encoder_outputs.last_hidden_state

        return audio_embedding

    def forward(self, text_labels):
        '''
        End-to-end: Text labels → Audio → Audio embeddings

        Args:
            text_labels (list of str) - Batch of text transcriptions

        Returns:
            audio_embeddings (torch.Tensor) - [batch, seq_len, audio_embedding_dim]
        '''
        batch_embeddings = []

        for text in text_labels:
            audio_embedding = self.text_to_audio_embedding(text)
            batch_embeddings.append(audio_embedding.squeeze(0))

        # Pad to max sequence length for embeddings
        max_seq_len = max(emb.size(0) for emb in batch_embeddings)
        embedding_dim = batch_embeddings[0].size(-1)
        batch_size = len(batch_embeddings)

        audio_embeddings = torch.zeros(
            batch_size, max_seq_len, embedding_dim,
            dtype=batch_embeddings[0].dtype,
            device=batch_embeddings[0].device
        )

        attention_mask = torch.zeros(
            batch_size, max_seq_len,
            dtype=torch.long,
            device=batch_embeddings[0].device
        )

        for i, emb in enumerate(batch_embeddings):
            seq_len = emb.size(0)
            audio_embeddings[i, :seq_len, :] = emb
            attention_mask[i, :seq_len] = 1
        

        return audio_embeddings, attention_mask
