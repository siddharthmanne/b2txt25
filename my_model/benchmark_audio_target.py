"""
Benchmark script comparing original AudioTarget vs AudioTargetOptimized
"""

import torch
import time
import sys
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Import both versions
from audio_target import AudioTarget as AudioTargetOriginal
from audio_target_optimized import AudioTargetOptimized

def benchmark_audio_target(batch_size=4, num_runs=3):
    """
    Benchmark both implementations

    Args:
        batch_size: Number of texts to process
        num_runs: Number of times to run each test
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print("="*70)

    # Test data - varying lengths
    test_texts = [
        "Hello world",
        "This is a longer sentence for testing purposes",
        "Short text",
        "Another example with different length and content for variety",
        "Testing speech synthesis",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing with neural networks"
    ]

    # Take subset based on batch_size
    texts = test_texts[:batch_size]

    print(f"\nTest texts ({len(texts)}):")
    for i, text in enumerate(texts):
        print(f"  {i+1}. '{text}'")

    # ========================================
    # Load shared models
    # ========================================
    print("\n" + "="*70)
    print("Loading shared Qwen2Audio model...")
    print("="*70)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # ========================================
    # Test 1: Original Implementation
    # ========================================
    print("\n" + "="*70)
    print("Test 1: Original AudioTarget (sequential)")
    print("="*70)

    audio_target_orig = AudioTargetOriginal(
        shared_a2t_model=model,
        shared_processor=processor,
        device=device
    )

    orig_times = []
    for run in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time.time()
        embeddings_orig, mask_orig = audio_target_orig(texts)
        elapsed = time.time() - start
        orig_times.append(elapsed)

        print(f"  Run {run+1}: {elapsed:.3f}s")
        if run == 0:
            print(f"    Output shape: {embeddings_orig.shape}")
            print(f"    Mask shape: {mask_orig.shape}")

    avg_orig_time = sum(orig_times) / len(orig_times)
    print(f"\n  Average time: {avg_orig_time:.3f}s")

    # ========================================
    # Test 2: Optimized Implementation
    # ========================================
    print("\n" + "="*70)
    print("Test 2: Optimized AudioTarget (parallel TTS + batch audio_tower)")
    print("="*70)

    audio_target_opt = AudioTargetOptimized(
        shared_a2t_model=model,
        shared_processor=processor,
        device=device,
        tts_num_workers=4
    )

    opt_times = []
    for run in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time.time()
        embeddings_opt, mask_opt = audio_target_opt(texts)
        elapsed = time.time() - start
        opt_times.append(elapsed)

        print(f"  Run {run+1}: {elapsed:.3f}s")
        if run == 0:
            print(f"    Output shape: {embeddings_opt.shape}")
            print(f"    Mask shape: {mask_opt.shape}")

    avg_opt_time = sum(opt_times) / len(opt_times)
    print(f"\n  Average time: {avg_opt_time:.3f}s")

    # ========================================
    # Verify outputs are equivalent
    # ========================================
    print("\n" + "="*70)
    print("Verification: Checking if outputs are equivalent")
    print("="*70)

    # Check shapes
    assert embeddings_orig.shape == embeddings_opt.shape, \
        f"Shape mismatch: {embeddings_orig.shape} vs {embeddings_opt.shape}"
    assert mask_orig.shape == mask_opt.shape, \
        f"Mask shape mismatch: {mask_orig.shape} vs {mask_opt.shape}"

    print("  ✓ Shapes match")

    # Check if embeddings are close (allowing for small numerical differences)
    # Note: Due to potential differences in TTS randomness or processing order,
    # embeddings might not be exactly identical
    try:
        # Check masks are identical
        mask_match = torch.equal(mask_orig, mask_opt)
        print(f"  Attention masks identical: {mask_match}")

        # Check embeddings similarity (using masked positions only)
        orig_valid = embeddings_orig[mask_orig.bool()]
        opt_valid = embeddings_opt[mask_opt.bool()]

        if orig_valid.shape == opt_valid.shape:
            # Compute similarity metrics
            mse = torch.mean((orig_valid - opt_valid) ** 2).item()
            max_diff = torch.max(torch.abs(orig_valid - opt_valid)).item()

            print(f"  MSE between valid embeddings: {mse:.6f}")
            print(f"  Max absolute difference: {max_diff:.6f}")

            # Embeddings should be very similar (not identical due to potential TTS randomness)
            if mse < 1e-4:
                print("  ✓ Embeddings are nearly identical")
            elif mse < 1e-2:
                print("  ⚠️  Embeddings have small differences (expected due to TTS)")
            else:
                print("  ⚠️  Embeddings have larger differences - investigate")
        else:
            print(f"  ⚠️  Different number of valid positions: {orig_valid.shape} vs {opt_valid.shape}")

    except Exception as e:
        print(f"  ⚠️  Could not compare embeddings: {e}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Original (sequential):    {avg_orig_time:.3f}s")
    print(f"  Optimized (parallelized): {avg_opt_time:.3f}s")
    print(f"  Speedup:                  {avg_orig_time/avg_opt_time:.2f}x")
    print("="*70)

    if avg_opt_time < avg_orig_time:
        improvement = (1 - avg_opt_time/avg_orig_time) * 100
        print(f"  ✓ Optimized version is {improvement:.1f}% faster!")
    else:
        print(f"  ⚠️  Optimized version is slower - check implementation")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if avg_orig_time/avg_opt_time > 1.2:
        print("  → Use AudioTargetOptimized for training (significant speedup)")
    elif avg_orig_time/avg_opt_time > 1.05:
        print("  → Use AudioTargetOptimized for training (modest speedup)")
    else:
        print("  → Speedup is minimal - parallelization overhead may dominate")
        print("  → Consider using original for simplicity, or test with larger batches")

    print("\n" + "="*70)
    print("SCALING TEST: How does speedup change with batch size?")
    print("="*70)
    print("  Run this script with different batch_size values:")
    print("    python benchmark_audio_target.py 2")
    print("    python benchmark_audio_target.py 4")
    print("    python benchmark_audio_target.py 8")
    print("    python benchmark_audio_target.py 16")
    print("\n  Typically, larger batches → better parallelization benefit")
    print("="*70)


if __name__ == "__main__":
    # Allow batch_size to be specified as command line argument
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    benchmark_audio_target(batch_size=batch_size, num_runs=num_runs)
