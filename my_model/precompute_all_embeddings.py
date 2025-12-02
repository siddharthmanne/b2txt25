"""
Standalone script to precompute audio embeddings for all transcriptions in the dataset.

This script:
1. Loads the AudioTarget model (TTS + AudioTower)
2. Scans all HDF5 dataset files to collect unique transcriptions
3. Computes audio embeddings for each unique transcription
4. Saves embeddings to a single HDF5 file for fast retrieval during training

The embeddings are stored in: cache/audio_embeddings/embeddings.h5
Structure:
    embeddings/{text_hash}/embedding - [seq_len, 1280]
    embeddings/{text_hash}/attention_mask - [seq_len]
    metadata/{text_hash} - attributes: 'text', 'seq_len'

Usage:
    python precompute_all_embeddings.py [--config training_args.yaml] [--force-recompute]

Arguments:
    --config: Path to training config file (default: training_args.yaml)
    --force-recompute: Recompute all embeddings even if already cached
"""

import torch
import sys
import os
import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import h5py
import numpy as np
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline
import librosa


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def text_to_hash(text):
    """Convert text to SHA256 hash for use as key."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_audio_models(args, device, logger):
    """
    Load the TTS and Audio Tower models.

    Args:
        args: Configuration dictionary
        device: Device to load models on
        logger: Logger instance

    Returns:
        tuple: (t2a_pipeline, a2t_model, processor, audio_tower, target_sr)
    """
    logger.info("=" * 80)
    logger.info("LOADING AUDIO MODELS")
    logger.info("=" * 80)

    # Load TTS model
    t2a_model_id = args['model']['t2a_model_id']
    logger.info(f"Loading TTS model: {t2a_model_id}")
    t2a_pipeline = pipeline("text-to-speech", t2a_model_id, device=device)

    # Load Audio-LLM model
    a2t_model_id = args['model']['a2t_model_id']
    logger.info(f"Loading Audio-LLM model: {a2t_model_id}")

    if args['model'].get('use_quantization', False):
        quantization_bits = args['model'].get('quantization_bits', 8)
        logger.info(f"Using {quantization_bits}-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(quantization_bits == 4),
            load_in_8bit=(quantization_bits == 8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            a2t_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            a2t_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(a2t_model_id)

    # Extract audio tower and target sampling rate
    audio_tower = a2t_model.audio_tower
    target_sr = processor.feature_extractor.sampling_rate

    # Freeze audio tower
    for param in audio_tower.parameters():
        param.requires_grad = False
    audio_tower.eval()

    logger.info(f"✓ Models loaded successfully")
    logger.info(f"  Target sampling rate: {target_sr} Hz")
    logger.info("=" * 80)

    return t2a_pipeline, a2t_model, processor, audio_tower, target_sr


def collect_unique_texts_from_files(file_paths, logger):
    """
    Collect all unique transcription texts from HDF5 dataset files.

    Args:
        file_paths: List of paths to HDF5 files
        logger: Logger instance

    Returns:
        List of unique text strings
    """
    logger.info("=" * 80)
    logger.info("SCANNING DATASET FILES FOR UNIQUE TRANSCRIPTIONS")
    logger.info("=" * 80)

    unique_texts = set()
    total_transcriptions = 0

    logger.info(f"Scanning {len(file_paths)} dataset files...")

    for file_path in tqdm(file_paths, desc="Scanning files"):
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        try:
            # Optimize HDF5 for network filesystems with large cache
            # rdcc_nbytes: chunk cache size (100 MB)
            # rdcc_nslots: number of cache slots (10000)
            with h5py.File(file_path, 'r', rdcc_nbytes=100*1024*1024, rdcc_nslots=10000) as f:
                # Get all trial keys at once
                trial_keys = [key for key in f.keys() if key.startswith('trial_')]

                # Batch read all transcriptions from this file for better network I/O
                transcriptions_batch = []
                for trial_key in trial_keys:
                    if 'transcription' in f[trial_key]:
                        transcriptions_batch.append((trial_key, f[trial_key]['transcription'][:]))

                # Now decode all transcriptions (fast, no I/O)
                for trial_key, trans_array in transcriptions_batch:
                    total_transcriptions += 1
                    try:
                        # Decode int32 array as ASCII characters
                        # Filter out null characters (0) and decode
                        text = ''.join([chr(c) for c in trans_array if c != 0])
                        text = text.strip()

                        if text:  # Only add non-empty texts
                            unique_texts.add(text)
                    except Exception as e:
                        logger.warning(f"Error decoding transcription in {file_path}/{trial_key}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

    unique_texts_list = sorted(list(unique_texts))

    logger.info("=" * 80)
    logger.info(f"✓ Scanning complete")
    logger.info(f"  Total transcriptions: {total_transcriptions}")
    logger.info(f"  Unique texts: {len(unique_texts_list)}")
    logger.info("=" * 80)

    return unique_texts_list


def compute_audio_embedding(
    text,
    t2a_pipeline,
    audio_tower,
    processor,
    a2t_model,
    target_sr,
    device
):
    """
    Compute audio embedding for a single text.

    Args:
        text: Text string
        t2a_pipeline: TTS pipeline
        audio_tower: Audio encoder model
        processor: Audio processor
        a2t_model: Full audio model (for dtype)
        target_sr: Target sampling rate
        device: Device to compute on

    Returns:
        tuple: (embedding, attention_mask)
            - embedding: [seq_len, 1280]
            - attention_mask: [seq_len]
    """
    with torch.no_grad():
        # Step 1: Text to audio via TTS
        tts_output = t2a_pipeline(text)
        audio_array = tts_output['audio']

        # Handle multi-channel audio
        if audio_array.ndim > 1:
            audio_array = audio_array[0]

        source_sr = tts_output['sampling_rate']

        # Step 2: Resample if needed
        if source_sr != target_sr:
            audio_array = audio_array.astype(np.float32)
            audio_array = librosa.resample(
                audio_array,
                orig_sr=source_sr,
                target_sr=target_sr
            )

        # Step 3: Process audio to mel-spectrogram
        inputs = processor(
            text=["<|AUDIO|>"],
            audios=[audio_array],
            sampling_rate=target_sr,
            return_tensors="pt"
        ).to(device)

        # Step 4: Encode to embeddings
        features = inputs.input_features.to(dtype=a2t_model.dtype)
        encoder_outputs = audio_tower(
            input_features=features,
            return_dict=True
        )
        embedding = encoder_outputs.last_hidden_state  # [1, seq_len, 1280]

        # Step 5: Create attention mask
        num_frames = inputs.input_features.size(-1)
        seq_len = num_frames // 2  # stride-2 pooling
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

        # Remove batch dimension
        embedding = embedding.squeeze(0)  # [seq_len, 1280]
        attention_mask = attention_mask.squeeze(0)  # [seq_len]

        return embedding, attention_mask


def initialize_embeddings_file(embeddings_path):
    """Create empty HDF5 file with proper structure."""
    with h5py.File(embeddings_path, 'w') as f:
        f.create_group('embeddings')
        f.create_group('metadata')


def check_if_cached(embeddings_path, text):
    """Check if text is already in embeddings file."""
    if not os.path.exists(embeddings_path):
        return False

    text_hash = text_to_hash(text)
    with h5py.File(embeddings_path, 'r') as f:
        return text_hash in f['embeddings']


def save_embedding(embeddings_path, text, embedding, attention_mask):
    """Save a single embedding to the HDF5 file."""
    text_hash = text_to_hash(text)

    # Convert to numpy
    embedding_np = embedding.detach().cpu().numpy()
    attention_mask_np = attention_mask.detach().cpu().numpy()

    with h5py.File(embeddings_path, 'a') as f:
        # Remove if exists
        if text_hash in f['embeddings']:
            del f['embeddings'][text_hash]
            del f['metadata'][text_hash]

        # Create groups
        emb_group = f['embeddings'].create_group(text_hash)
        meta_group = f['metadata'].create_group(text_hash)

        # Save data
        emb_group.create_dataset('embedding', data=embedding_np, compression='gzip')
        emb_group.create_dataset('attention_mask', data=attention_mask_np, compression='gzip')

        # Save metadata
        meta_group.attrs['text'] = text
        meta_group.attrs['seq_len'] = embedding_np.shape[0]
        meta_group.attrs['embedding_dim'] = embedding_np.shape[1]


def precompute_embeddings(
    texts,
    embeddings_path,
    t2a_pipeline,
    audio_tower,
    processor,
    a2t_model,
    target_sr,
    device,
    logger,
    force_recompute=False
):
    """
    Precompute embeddings for all texts and save to HDF5 file.

    Args:
        texts: List of text strings
        embeddings_path: Path to HDF5 embeddings file
        t2a_pipeline: TTS pipeline
        audio_tower: Audio encoder
        processor: Audio processor
        a2t_model: Full model (for dtype)
        target_sr: Target sampling rate
        device: Device to compute on
        logger: Logger instance
        force_recompute: Recompute even if cached
    """
    logger.info("=" * 80)
    logger.info("PRECOMPUTING AUDIO EMBEDDINGS")
    logger.info("=" * 80)

    # Create embeddings file if doesn't exist
    if not os.path.exists(embeddings_path):
        initialize_embeddings_file(embeddings_path)

    # Filter out already cached texts
    if force_recompute:
        texts_to_compute = texts
        logger.info(f"Force recompute enabled - processing all {len(texts_to_compute)} texts")
    else:
        texts_to_compute = [text for text in texts if not check_if_cached(embeddings_path, text)]
        n_cached = len(texts) - len(texts_to_compute)
        logger.info(f"Found {len(texts)} unique texts")
        logger.info(f"  Already cached: {n_cached}")
        logger.info(f"  Need to compute: {len(texts_to_compute)}")

    if len(texts_to_compute) == 0:
        logger.info("✓ All texts already cached!")
        return

    logger.info(f"Starting computation...")

    # Compute embeddings with progress bar
    for i, text in enumerate(tqdm(texts_to_compute, desc="Computing embeddings")):
        try:
            embedding, attention_mask = compute_audio_embedding(
                text=text,
                t2a_pipeline=t2a_pipeline,
                audio_tower=audio_tower,
                processor=processor,
                a2t_model=a2t_model,
                target_sr=target_sr,
                device=device
            )

            # Save to file
            save_embedding(embeddings_path, text, embedding, attention_mask)

            # Log progress every 100 texts
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(texts_to_compute)} texts")

        except Exception as e:
            logger.error(f"Error computing embedding for text '{text[:50]}...': {e}")
            continue

    logger.info("=" * 80)
    logger.info(f"✓ Successfully precomputed {len(texts_to_compute)} embeddings")
    logger.info("=" * 80)


def print_file_stats(embeddings_path, logger):
    """Print statistics about the embeddings file."""
    if not os.path.exists(embeddings_path):
        logger.warning("Embeddings file does not exist")
        return

    try:
        with h5py.File(embeddings_path, 'r') as f:
            n_entries = len(f['embeddings'].keys())
            file_size_mb = os.path.getsize(embeddings_path) / (1024 * 1024)

            # Sample to get average seq_len
            avg_seq_len = 0
            if n_entries > 0:
                sample_size = min(100, n_entries)
                seq_lens = []
                for i, key in enumerate(f['embeddings'].keys()):
                    if i >= sample_size:
                        break
                    seq_lens.append(f['metadata'][key].attrs['seq_len'])
                avg_seq_len = np.mean(seq_lens)

            logger.info("=" * 80)
            logger.info("EMBEDDINGS FILE STATISTICS")
            logger.info("=" * 80)
            logger.info(f"  Cached entries:     {n_entries}")
            logger.info(f"  File size:          {file_size_mb:.2f} MB")
            logger.info(f"  Avg sequence len:   {avg_seq_len:.1f}")
            logger.info(f"  File location:      {embeddings_path}")
            logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error reading stats: {e}")


def main():
    """Main function to precompute all embeddings."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Precompute audio embeddings for all dataset transcriptions")
    parser.add_argument('--config', type=str, default='training_args.yaml', help='Path to training config file')
    parser.add_argument('--force-recompute', action='store_true', help='Recompute all embeddings even if cached')
    parser.add_argument('--cache-dir', type=str, default=None, help='Override cache directory from config')
    args_cli = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("AUDIO EMBEDDING PRECOMPUTATION SCRIPT")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading config from: {args_cli.config}")
    args = OmegaConf.load(args_cli.config)

    # Set cache directory
    if args_cli.cache_dir:
        cache_dir = args_cli.cache_dir
    else:
        cache_dir = args.get('cache_dir', 'cache/audio_embeddings')

    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    embeddings_path = os.path.join(cache_dir, 'embeddings.h5')

    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Embeddings file: {embeddings_path}")

    # Print initial stats
    print_file_stats(embeddings_path, logger)

    # Setup device
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires GPU.")
        sys.exit(1)

    device = torch.device("cuda:0")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # Load audio models
    t2a_pipeline, a2t_model, processor, audio_tower, target_sr = load_audio_models(args, device, logger)

    # Collect dataset file paths from BOTH train and val sets
    dataset_dir = args['dataset']['dataset_dir']
    sessions = args['dataset']['sessions']

    # Only include files that actually exist
    train_files = []
    val_files = []
    for s in sessions:
        train_path = os.path.join(dataset_dir, s, 'data_train.hdf5')
        val_path = os.path.join(dataset_dir, s, 'data_val.hdf5')

        if os.path.exists(train_path):
            train_files.append(train_path)
        else:
            logger.warning(f"Missing train file for session {s}")

        if os.path.exists(val_path):
            val_files.append(val_path)
        # No warning for missing val files - some sessions don't have validation data

    all_files = train_files + val_files  # Combine both train and val files

    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Number of sessions: {len(sessions)}")
    logger.info(f"Total dataset files: {len(all_files)} ({len(train_files)} train + {len(val_files)} val)")
    if len(val_files) < len(sessions):
        logger.info(f"  Note: {len(sessions) - len(val_files)} sessions don't have validation files (expected)")

    # Collect unique texts from all files (train + val)
    unique_texts = collect_unique_texts_from_files(all_files, logger)

    # Precompute embeddings
    precompute_embeddings(
        texts=unique_texts,
        embeddings_path=embeddings_path,
        t2a_pipeline=t2a_pipeline,
        audio_tower=audio_tower,
        processor=processor,
        a2t_model=a2t_model,
        target_sr=target_sr,
        device=device,
        logger=logger,
        force_recompute=args_cli.force_recompute
    )

    # Print final stats
    print_file_stats(embeddings_path, logger)

    logger.info("✓ Precomputation complete!")
    logger.info(f"Embeddings saved at: {embeddings_path}")
    logger.info("")
    logger.info("You can now use these embeddings during training by setting:")
    logger.info("  use_audio_cache: true")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info("in your training_args.yaml")


if __name__ == "__main__":
    main()
