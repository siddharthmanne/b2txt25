import torch
from torch import nn
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

class LLMDecoder(nn.Module):
    """
    Frozen LLM decoder for text generation from aligned brain embeddings.

    Pipeline: aligned_brain_embedding → Qwen Projector (frozen) → LLM (frozen) → text

    Responsibilities:
    1. Accept aligned_brain_embedding from BrainEncoder
    2. Pass through Qwen's multi_modal_projector (frozen)
    3. Build input sequence for LLM with text prompt embeddings
    4. Generate text autoregressively or compute loss during training
    """
    def __init__(
        self,
        a2t_model_id="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        freeze_all=True
    ):
        '''
        a2t_model_id (str) - Qwen Audio-to-Text model ID
        device (str)       - Device to run on
        freeze_all (bool)  - Whether to freeze all parameters
        '''
        super(LLMDecoder, self).__init__()

        self.device = device
        self.a2t_model_id = a2t_model_id

        # Load Qwen Audio-to-Text model
        self.a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            a2t_model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # Extract components
        self.audio_processor = AutoProcessor.from_pretrained(a2t_model_id)
        self.projector = self.a2t_model.multi_modal_projector  # Audio emb → LLM emb
        self.llm_decoder = self.a2t_model.language_model       # The actual LLM

        # Get tokenizer
        self.tokenizer = self.audio_processor.tokenizer

        # Get LLM components
        self.embedding_layer = self.llm_decoder.get_input_embeddings()
        self.d_model = self.llm_decoder.config.hidden_size
        self.vocab_size = self.llm_decoder.config.vocab_size

        # Freeze everything if specified
        if freeze_all:
            for param in self.projector.parameters():
                param.requires_grad = False
            for param in self.llm_decoder.parameters():
                param.requires_grad = False

            self.projector.eval()
            self.llm_decoder.eval()

    def project_to_llm_space(self, aligned_brain_embedding):
        '''
        Project aligned brain embeddings to LLM embedding space

        Args:
            aligned_brain_embedding: [batch, time, audio_embedding_dim]

        Returns:
            llm_embeddings: [batch, time, d_model]
        '''
        with torch.no_grad():
            llm_embeddings = self.projector(aligned_brain_embedding)
        return llm_embeddings

    def build_train_batch(self, brain_prefix, target_texts):
        '''
        Build training batch with brain prefix + text embeddings

        Args:
            brain_prefix: [batch, T_brain, d_model] - Projected brain embeddings
            target_texts: list of str - Ground truth transcriptions

        Returns:
            inputs_embeds: [batch, T_brain + T_text, d_model]
            labels: [batch, T_brain + T_text] - Loss computed only on text part
        '''
        batch_size, T_brain, _ = brain_prefix.shape

        # 1. Tokenize target text with prompt
        prompts = [f"Transcript: {t}" for t in target_texts]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        input_ids = enc.input_ids       # [batch, T_text]
        attn_mask = enc.attention_mask  # [batch, T_text]

        # 2. Convert tokens to embeddings
        with torch.no_grad():
            text_embs = self.embedding_layer(input_ids)  # [batch, T_text, d_model]

        # 3. Concatenate brain_prefix + text_embs
        inputs_embeds = torch.cat([brain_prefix, text_embs], dim=1)

        # 4. Build labels: -100 for brain part (ignore), actual tokens for text part
        ignore = torch.full(
            (batch_size, T_brain),
            fill_value=-100,
            device=self.device,
            dtype=torch.long
        )
        labels = torch.cat([ignore, input_ids], dim=1)

        return inputs_embeds, labels

    def forward_train(self, aligned_brain_embedding, target_texts):
        '''
        Training forward pass: Compute LLM loss

        Args:
            aligned_brain_embedding: [batch, time, audio_embedding_dim]
            target_texts: list of str

        Returns:
            llm_loss: Cross-entropy loss on text prediction
            logits: [batch, seq_len, vocab_size]
        '''
        # Project brain embeddings to LLM space
        brain_prefix = self.project_to_llm_space(aligned_brain_embedding)

        # Build input batch
        inputs_embeds, labels = self.build_train_batch(brain_prefix, target_texts)

        # Forward through LLM (frozen, but still computes gradients for brain_encoder)
        outputs = self.llm_decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True
        )

        return outputs.loss, outputs.logits

    def generate(self, aligned_brain_embedding, max_length=50):
        '''
        Inference: Generate text autoregressively

        Args:
            aligned_brain_embedding: [batch, time, audio_embedding_dim]
            max_length: Maximum sequence length

        Returns:
            generated_texts: list of str
        '''
        # Project to LLM space
        brain_prefix = self.project_to_llm_space(aligned_brain_embedding)

        # Build prompt
        prompt = "Transcript: "
        enc = self.tokenizer(
            [prompt] * brain_prefix.shape[0],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            prompt_embs = self.embedding_layer(enc.input_ids)

        # Concatenate brain + prompt
        inputs_embeds = torch.cat([brain_prefix, prompt_embs], dim=1)

        # Generate
        with torch.no_grad():
            outputs = self.llm_decoder.generate(
                inputs_embeds=inputs_embeds,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )

        # Decode to text
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts
