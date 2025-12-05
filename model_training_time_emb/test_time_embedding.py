"""
Test script to verify time embedding implementation
"""
import torch
import sys
from dataset import get_sinusoidal_time_embedding, session_to_day_index

def test_sinusoidal_embedding():
    """Test sinusoidal time embedding generation"""
    print("Testing sinusoidal time embedding generation...")

    # Test different day indices
    day_indices = [0, 2, 7, 14, 30]
    embedding_dim = 32

    print(f"\nEmbedding dimension: {embedding_dim}")

    for day_idx in day_indices:
        emb = get_sinusoidal_time_embedding(day_idx, embedding_dim)
        print(f"Day {day_idx}: shape={emb.shape}, min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")

        # Verify shape
        assert emb.shape == (embedding_dim,), f"Expected shape ({embedding_dim},), got {emb.shape}"

    # Test that different days produce different embeddings
    emb0 = get_sinusoidal_time_embedding(0, embedding_dim)
    emb7 = get_sinusoidal_time_embedding(7, embedding_dim)

    # Embeddings should be different
    assert not torch.allclose(emb0, emb7), "Embeddings for different days should be different"
    print("\n✓ Sinusoidal embeddings are unique for different days")

    # Test that embeddings are deterministic
    emb0_copy = get_sinusoidal_time_embedding(0, embedding_dim)
    assert torch.allclose(emb0, emb0_copy), "Embeddings should be deterministic"
    print("✓ Sinusoidal embeddings are deterministic")

    print("\n✓ All sinusoidal embedding tests passed!")

def test_session_to_day_index():
    """Test session name to day index conversion"""
    print("\nTesting session name to day index conversion...")

    # Test cases based on the config file sessions
    test_cases = [
        ('t15.2023.08.11', 0),   # Reference date
        ('t15.2023.08.13', 2),   # 2 days later
        ('t15.2023.08.18', 7),   # 7 days later
        ('t15.2023.08.20', 9),   # 9 days later
        ('t15.2023.09.01', 21),  # 21 days later
    ]

    for session_name, expected_day in test_cases:
        day_idx = session_to_day_index(session_name)
        print(f"Session {session_name}: day index = {day_idx} (expected {expected_day})")
        assert day_idx == expected_day, f"Expected {expected_day}, got {day_idx}"

    print("\n✓ All session to day index tests passed!")

def test_embedding_concatenation():
    """Test that embeddings can be concatenated with neural features"""
    print("\nTesting embedding concatenation with neural features...")

    # Simulate neural features
    batch_size = 4
    num_time_steps = 100
    neural_dim = 512
    time_emb_dim = 32

    # Create dummy neural features
    neural_features = torch.randn(batch_size, num_time_steps, neural_dim)

    # Create time embeddings for each sample
    time_embeddings = []
    for i in range(batch_size):
        day_idx = i * 2  # Use different days for each batch item
        time_emb = get_sinusoidal_time_embedding(day_idx, time_emb_dim)
        # Expand to match time steps
        time_emb_expanded = time_emb.unsqueeze(0).expand(num_time_steps, -1)
        time_embeddings.append(time_emb_expanded)

    # Stack time embeddings
    time_embeddings = torch.stack(time_embeddings, dim=0)

    print(f"Neural features shape: {neural_features.shape}")
    print(f"Time embeddings shape: {time_embeddings.shape}")

    # Concatenate
    combined = torch.cat([neural_features, time_embeddings], dim=2)

    print(f"Combined features shape: {combined.shape}")

    expected_shape = (batch_size, num_time_steps, neural_dim + time_emb_dim)
    assert combined.shape == expected_shape, f"Expected shape {expected_shape}, got {combined.shape}"

    print(f"\n✓ Successfully concatenated embeddings!")
    print(f"  Input: [{batch_size}, {num_time_steps}, {neural_dim}]")
    print(f"  Output: [{batch_size}, {num_time_steps}, {neural_dim + time_emb_dim}]")

if __name__ == '__main__':
    print("=" * 60)
    print("Time Embedding Implementation Tests")
    print("=" * 60)

    try:
        test_sinusoidal_embedding()
        test_session_to_day_index()
        test_embedding_concatenation()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
