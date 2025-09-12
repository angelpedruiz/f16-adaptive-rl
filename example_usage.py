"""
Example usage script demonstrating how to use the refactored training system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    """
    Example showing how to run training with different configurations.
    """
    
    print("F16 Adaptive RL Training System Examples")
    print("=" * 60)
    
    print("\nAvailable configurations:")
    configs_dir = Path("configs")
    if configs_dir.exists():
        for config_file in configs_dir.glob("*.yaml"):
            print(f"  - {config_file.name}")
    
    print("\nUsage examples:")
    print("1. Train ADHDP agent:")
    print("   python train.py --config configs/adhdp_config.yaml")
    
    print("\n2. Train Q-Learning agent:")
    print("   python train.py --config configs/q_learning_config.yaml")
    
    print("\n3. Train DQN agent:")
    print("   python train.py --config configs/dqn_config.yaml")
    
    print("\n4. Resume training from checkpoint:")
    print("   python train.py --config configs/adhdp_config.yaml --resume-from experiments/adhdp/run_123/checkpoint_ep1000")
    
    print("\nKey features:")
    print("  + Fully config-driven training")
    print("  + Automatic checkpointing and resuming")
    print("  + Training metrics and trajectory plotting")
    print("  + Modular agent and environment system")
    print("  + RecordEpisodeStatistics integration")
    print("  + Unique run directories with timestamps")
    
    print("\nConfiguration structure:")
    print("  - agent: Agent type and hyperparameters")
    print("  - environment: Environment settings and reference signals")
    print("  - training: Episode count, seed, logging frequency")
    print("  - checkpointing: Save intervals and resume options")
    print("  - plotting: Metrics and trajectory visualization")
    print("  - evaluation: Performance analysis settings")
    
    print(f"\n{'-'*60}")
    print("Ready to train! Use the commands above to get started.")


if __name__ == "__main__":
    main()