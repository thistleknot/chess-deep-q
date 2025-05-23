import numpy as np
import torch
import random
import chess
import os
import sys

# ADD THESE MISSING IMPORTS:
import logging
import functools
import traceback
from datetime import datetime

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
    # Set all random seeds here
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
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

import traceback

# STEP 2: ADD this decorator function (but keep your existing functions):
def comprehensive_error_logger(func):
    """
    Decorator that sets up comprehensive logging for the entire application.
    Catches ALL errors that bubble up and logs them to a file.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Set up logging with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/chess_ai_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        
        # Custom exception handler to log uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            print(f"\n{'='*60}")
            print("CRITICAL ERROR LOGGED!")
            print(f"Check log file: {log_filename}")
            print(f"Error: {exc_value}")
            print(f"{'='*60}")
        
        # Set the custom exception handler
        sys.excepthook = handle_exception
        
        # Log startup
        logger.info("="*60)
        logger.info("CHESS AI APPLICATION STARTED")
        logger.info(f"Log file: {log_filename}")
        logger.info("="*60)
        
        try:
            logger.info("Starting main application function...")
            result = func(*args, **kwargs)
            logger.info("Main application function completed successfully")
            return result
            
        except Exception as e:
            logger.critical(f"Exception in main function: {e}")
            logger.critical("Full traceback:")
            logger.critical(traceback.format_exc())
            
            print(f"\n{'='*60}")
            print("MAIN FUNCTION ERROR!")
            print(f"Error: {e}")
            print(f"Full details logged to: {log_filename}")
            print(f"{'='*60}")
            
            raise
            
        finally:
            logger.info("Chess AI application session ended")
            logger.info("="*60)
    
    return wrapper

# STEP 3: MODIFY your existing main() function by adding just ONE line:
@comprehensive_error_logger  # <-- ADD THIS LINE
def main():
    """Main entry point for the chess AI application"""
    # KEEP ALL YOUR EXISTING CODE EXACTLY AS IS
    try:
        # Setup the environment
        device = setup_environment()  # <-- This calls YOUR existing function
        
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