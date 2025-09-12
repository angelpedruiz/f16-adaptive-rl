#!/usr/bin/env python3
"""
Quick performance test script to benchmark the new training loop.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    """Run performance test with minimal config."""
    print("🔍 Running performance test...")
    print("Testing training loop performance...")
    
    start_time = time.time()
    
    try:
        from train import train_agent
        train_agent("configs/speed_test.yaml")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return
    
    total_time = time.time() - start_time
    print(f"\n✅ Performance test completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()