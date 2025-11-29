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
        shared_a2t_model=None,
        shared_processor=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        freeze_all=True
    ):
        '''
        shared_a2t_model         - Shared Qwen2AudioForConditionalGeneration instance
        shared_processor         - Shared AutoProcessor instance
        device (str)             - Device to run on
        freeze_all (bool)        - Whether to freeze all parameters
        '''
        super(LLMDecoder, self).__init__()

        self.device = device

        # Use shared model and processor (passed from Brain2TextModel)
        if shared_a2t_model is None or shared_processor is None:
            raise ValueError("LLMDecoder requires shared_a2t_model and shared_processor to be provided")

        self.a2t_model = shared_a2t_model
        self.audio_processor = shared_processor

        # Extract components
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
        Project aligned brain embeddings to LLM embedding space using multi_modal_projector

        Args:
            aligned_brain_embedding: [batch, seq_len, 1280] - brain embeddings aligned to audio space

        Returns:
            llm_embeddings: [batch, seq_len, 3584] - projected to Qwen2-7B LLM space
        '''
        # Multi-modal projector: Linear(1280, 3584) - maps audio encoder dim to LLM dim
        with torch.no_grad():
            llm_embeddings = self.projector(aligned_brain_embedding)  # [batch, seq_len, 3584]
        return llm_embeddings  # [batch, seq_len, 3584]

    def build_train_batch(self, brain_prefix, target_texts):
        '''
        Build training batch for causal language modeling with brain prefix

        ═══════════════════════════════════════════════════════════════════════
        CAUSAL LM TRAINING: DETAILED EXPLANATION
        ═══════════════════════════════════════════════════════════════════════

        Example: target_text = "hello world", tokenized as [1234, 5678, 9999] (EOS)

        STEP 1: Build input sequence
        -----------------------------
        inputs_embeds = [brain_1, ..., brain_T, emb(1234), emb(5678), emb(9999)]
        labels        = [-100,    ..., -100,    1234,      5678,      9999]

        STEP 2: LLM forward pass
        ------------------------
        LLM processes entire sequence and outputs logits at EVERY position:
        logits = LLM(inputs_embeds)  # shape: [batch, T_brain + T_text, vocab_size=156032]

        For each position i, logits[i] is a distribution over 156032 vocabulary tokens.

        STEP 3: Loss computation (automatic in HuggingFace)
        ---------------------------------------------------
        PyTorch's cross_entropy loss does the following:

        For each position i where labels[i] != -100:
            1. Take logits[i-1]: [vocab_size] - raw scores over vocabulary
            2. Apply softmax: probs[i-1] = softmax(logits[i-1])
               - This converts raw scores to probabilities summing to 1
               - probs[i-1][k] = probability of token k at position i
            3. Target: labels[i] is a single token ID (e.g., 1234 for "hello")
               - This represents a one-hot vector: [0, 0, ..., 1, ..., 0]
               - probability = 1.0 at position labels[i], 0.0 elsewhere
            4. Cross-entropy: -log(probs[i-1][labels[i]])
               - Measures: "how much probability mass did model put on correct token?"
               - If probs[i-1][1234] = 0.8, loss = -log(0.8) = 0.22 (good!)
               - If probs[i-1][1234] = 0.01, loss = -log(0.01) = 4.6 (bad!)

        Concrete example:
        Position i-1: brain_T     (context: all brain embeddings)
        Position i:   label=1234  ("hello")
        → LLM predicts: logits[brain_T] = [0.1, ..., 3.2, ...] (raw scores)
        → After softmax: probs[brain_T][1234] = 0.15 (15% probability on "hello")
        → Loss: -log(0.15) = 1.90

        The labels array tells PyTorch:
        - Positions 0 to T_brain-1: labels = -100 → SKIP (no loss)
        - Positions T_brain to end: labels = token IDs → COMPUTE LOSS

        INFERENCE (different from training):
        -----------------------------------
        inputs_embeds = [brain_1, ..., brain_T, emb("Transcript:")]
        LLM generates autoregressively:
        1. Predict next token from logits[last_position]
        2. Sample/argmax to get token_1
        3. Append emb(token_1) and repeat

        Args:
            brain_prefix: [batch, T_brain, 3584] - Projected brain embeddings in LLM space
            target_texts: list of str - Ground truth transcriptions (e.g., ["hello world"])

        Returns:
            inputs_embeds: [batch, T_brain + T_text, 3584] - Full sequence embeddings
            labels: [batch, T_brain + T_text] - Token IDs for loss, -100 for masked positions
        '''
        batch_size, T_brain, _ = brain_prefix.shape  # T_brain = brain seq_len, _ = 3584

        # ===== Tokenize ground truth text with prompt =====
        prompts = [f"Transcript: {t}" for t in target_texts]  # Add prompt prefix
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        input_ids = enc.input_ids  # [batch, T_text] - token IDs including "Transcript: <text>"
        attn_mask = enc.attention_mask  # [batch, T_text]

        # ===== Convert tokens to embeddings =====
        with torch.no_grad():
            text_embs = self.embedding_layer(input_ids)  # [batch, T_text, 3584]

        # ===== Concatenate brain + text =====
        inputs_embeds = torch.cat([brain_prefix, text_embs], dim=1)  # [batch, T_brain + T_text, 3584]

        # ===== Build labels: -100 for brain, token IDs for text =====
        ignore = torch.full(
            (batch_size, T_brain),
            fill_value=-100,  # PyTorch CE loss ignores -100
            device=self.device,
            dtype=torch.long
        )  # [batch, T_brain]

        labels = torch.cat([ignore, input_ids], dim=1)  # [batch, T_brain + T_text]

        return inputs_embeds, labels  # [batch, T_brain + T_text, 3584], [batch, T_brain + T_text]

    def forward_train(self, aligned_brain_embedding, target_texts):
        '''
        Training forward pass: Compute LLM loss

        Args:
            aligned_brain_embedding: [batch, seq_len, 1280] - brain embeddings in audio space
            target_texts: list of str - ground truth transcriptions

        Returns:
            llm_loss: scalar - cross-entropy loss on text prediction
            logits: [batch, T_brain + T_text, 156032] - LLM output logits over vocabulary
        '''
        # Project brain embeddings to LLM space: 1280 → 3584
        brain_prefix = self.project_to_llm_space(aligned_brain_embedding)  # [batch, seq_len, 3584]

        # Build input batch with brain prefix + target text
        inputs_embeds, labels = self.build_train_batch(brain_prefix, target_texts)
        # inputs_embeds: [batch, T_brain + T_text, 3584]
        # labels: [batch, T_brain + T_text] with -100 for brain positions

        # Forward through LLM (frozen, but gradients flow back to brain_encoder)
        outputs = self.llm_decoder(
            inputs_embeds=inputs_embeds,  # [batch, T_brain + T_text, 3584]
            labels=labels,  # [batch, T_brain + T_text]
            return_dict=True
        )
        # outputs.logits: [batch, T_brain + T_text, 156032]
        # outputs.loss: scalar - average cross-entropy over non-masked positions

        return outputs.loss, outputs.logits  # scalar, [batch, T_brain + T_text, 156032]

    def generate(self, aligned_brain_embedding, max_length=50):
        '''
        Inference: Generate text autoregressively from brain embeddings

        Unlike training, we DON'T provide the ground truth text. Instead:
        1. Start with: [brain_embeddings] + ["Transcript:"]
        2. LLM predicts next token
        3. Append that token's embedding and repeat
        4. Stop when EOS token or max_length reached

        Args:
            aligned_brain_embedding: [batch, seq_len, 1280] - brain embeddings in audio space
            max_length: int - maximum tokens to generate

        Returns:
            generated_texts: list of str - decoded text predictions
        '''
        # Project brain embeddings to LLM space: 1280 → 3584
        brain_prefix = self.project_to_llm_space(aligned_brain_embedding)  # [batch, seq_len, 3584]

        # Build prompt: "Transcript: " to guide generation
        prompt = "Transcript: "
        batch_size = brain_prefix.shape[0]
        enc = self.tokenizer(
            [prompt] * batch_size,  # Repeat prompt for each batch item
            return_tensors="pt",
            padding=True
        ).to(self.device)
        # enc.input_ids: [batch, prompt_len] - usually just 2-3 tokens

        # Convert prompt tokens to embeddings
        with torch.no_grad():
            prompt_embs = self.embedding_layer(enc.input_ids)  # [batch, prompt_len, 3584]

        # Concatenate brain prefix + prompt embeddings
        inputs_embeds = torch.cat([brain_prefix, prompt_embs], dim=1)  # [batch, seq_len + prompt_len, 3584]

        # Generate text autoregressively
        # HuggingFace's generate() handles the autoregressive loop internally
        with torch.no_grad():
            outputs = self.llm_decoder.generate(
                inputs_embeds=inputs_embeds,  # [batch, seq_len + prompt_len, 3584]
                max_length=max_length,  # Max tokens to generate (total length)
                num_beams=5,  # Beam search for better quality
                early_stopping=True  # Stop at EOS token
            )
            # outputs: [batch, generated_length] - token IDs

        # Decode token IDs back to text
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # generated_texts: list of batch_size strings

        return generated_texts  # list of str
