#!/usr/bin/env python3
"""
Example: Running vLLM in embedding-only mode (text mode) with embedding inputs

This example demonstrates:
1. How to start the server with embedding-only mode for memory savings
2. How to create and send embedding inputs via the OpenAI API
"""

import json
import torch
from openai import OpenAI
from vllm.utils.serial_utils import tensor2base64

# ============================================================================
# STEP 1: Start the server with this command:
# ============================================================================
"""
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --enable-mm-embeds \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --port 8000
"""

# ============================================================================
# STEP 2: Create a client and example embedding
# ============================================================================

def create_example_image_embedding(
    num_images: int = 1,
    grid_size: int = 16,       # Grid size after spatial merge (16x16 = 256 features)
    hidden_size: int = 4096,   # Hidden size for Qwen2.5-VL models
    dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy image embedding tensor with grid metadata.

    For Qwen2.5-VL, which uses mRoPE (multi-dimensional RoPE), we need:
    - image_embeds: (num_images, feature_size, hidden_size)
    - image_grid_thw: (num_images, 3) containing [t, h, w] before spatial merge

    Args:
        num_images: Number of images
        grid_size: Grid size after spatial merge (default 16 for 16x16 grid = 256 features)
        hidden_size: Model hidden dimension
        dtype: Tensor dtype

    Returns:
        Tuple of (image_embeds, image_grid_thw)
    """
    # Calculate feature size: grid_size x grid_size
    # For a 16x16 grid after merge, we have 256 features
    feature_size = grid_size * grid_size

    # Create random embeddings (in practice, these would come from your encoder)
    image_embeds = torch.randn(
        num_images,
        feature_size,
        hidden_size,
        dtype=dtype
    )

    # Create grid dimensions: (t, h, w) BEFORE spatial merge
    # With spatial_merge_size=2 (default for Qwen2.5-VL):
    # - A 16x16 grid after merge corresponds to 32x32 before merge
    # - t=1 (temporal dimension, for images)
    spatial_merge_size = 2
    original_h = grid_size * spatial_merge_size
    original_w = grid_size * spatial_merge_size

    image_grid_thw = torch.tensor(
        [[1, original_h, original_w]] * num_images,
        dtype=torch.int64
    )

    return image_embeds, image_grid_thw


def main():
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",  # Not needed for local server
        base_url="http://localhost:8000/v1",
    )
    
    # Get the model name from the server (must match the model the server was started with)
    try:
        models = client.models.list()
        if models.data:
            model_name = models.data[0].id
            print(f"Using model from server: {model_name}")
        else:
            raise ValueError("No models found on server")
    except Exception as e:
        print(f"Warning: Could not get model from server: {e}")
        print("Using default model name. Make sure it matches your server.")
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # Fallback: must match server
    
    # Create example image embedding with grid metadata
    # For Qwen2.5-VL models, we need both embeddings and grid dimensions
    print("Creating example image embedding with grid metadata...")
    image_embeds, image_grid_thw = create_example_image_embedding(
        num_images=1,
        grid_size=16,      # 16x16 = 256 features after merge
        hidden_size=4096,
        dtype=torch.float16
    )
    print(f"Image embeddings shape: {image_embeds.shape}")
    print(f"Image grid_thw shape: {image_grid_thw.shape}")
    print(f"Grid dimensions (t, h, w): {image_grid_thw[0].tolist()}")

    # Convert to base64 for API transmission
    base64_image_embeds = tensor2base64(image_embeds)
    base64_image_grid_thw = tensor2base64(image_grid_thw)
    print("Embeddings and grid metadata serialized to base64")
    
    # Send request with embedding input
    print("\nSending chat completion request...")
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Describe it briefly."
                    },
                    {
                        "type": "image_embeds",
                        "image_embeds": {
                            "image_embeds": base64_image_embeds,
                            "image_grid_thw": base64_image_grid_thw,
                        },
                    },
                ],
            },
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    print("\n" + "="*60)
    print("Response:")
    print("="*60)
    print(chat_completion.choices[0].message.content)
    print("="*60)


if __name__ == "__main__":
    main()

