#!/usr/bin/env python3
"""
Simple test script to verify QwenSampler works correctly.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_evals.sampler.qwen_sampler import QwenSampler
from simple_evals.types import MessageList


def test_qwen_sampler():
    """Test basic QwenSampler functionality."""
    print("Initializing QwenSampler...")
    sampler = QwenSampler(
        model="Qwen/Qwen2.5-3B-Instruct",
        temperature=0.0,
        seed=42,
    )
    
    print("Testing simple text generation...")
    messages: MessageList = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]
    
    response = sampler(messages)
    print(f"Response: {response.response_text}")
    print(f"Metadata: {response.response_metadata}")
    
    print("\nTesting multi-turn conversation...")
    messages = [
        {"role": "user", "content": "Hello! What's your name?"},
    ]
    
    response = sampler(messages)
    print(f"Response: {response.response_text}")
    
    print("\n✅ QwenSampler test completed successfully!")


if __name__ == "__main__":
    test_qwen_sampler()
