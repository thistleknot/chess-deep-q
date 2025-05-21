import numpy as np
import torch
import random
import chess
import os
import sys

# Set matplotlib backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend which is interactive

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Ensure modules in the current directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main menu function
from menu import main as menu_main

def setup_environment():
    """Setup the environment for the chess AI"""
    # Check if required packages are installed
    required_packages = ['numpy', 'torch', 'chess', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA not available. Using CPU for computation.")
        print("Training may be slow. Consider enabling CUDA if available.")
    
    # Create directories for saved data if they don't exist
    os.makedirs("saved_games", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    return device

def main():
    """Main entry point for the chess AI application"""
    try:
        # Setup the environment
        device = setup_environment()
        
        # Display welcome message
        print("\n" + "="*78)
        print(" "*20 + "Chess AI with Deep Q-Learning and MCTS" + " "*20)
        print("="*78 + "\n")
        
        # Start the application
        menu_main()
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        
        # In debug mode, show the full traceback
        if os.environ.get('DEBUG', '0') == '1':
            import traceback
            traceback.print_exc()
        
    print("\nExiting Chess AI. Goodbye!")

if __name__ == "__main__":
    main()