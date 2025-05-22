import queue
import threading
import time
import chess
import matplotlib.pyplot as plt
from ui import NonClickableChessBoard
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend

class ThreadedChessBoard(NonClickableChessBoard):
    def __init__(self, board, ai, human_color=chess.WHITE):
        super().__init__(board, ai, human_color)
        self.ui_queue = queue.Queue()
        self.running = True
        self.ui_thread = None
        
    def _ui_thread_function(self):
        """Run matplotlib in its own thread"""
        plt.ion()  # Interactive mode
        
        # Initial board setup
        self.update_board()
        
        # If AI goes first (human is black), let AI make first move
        if self.human_color == chess.BLACK:
            # Signal the main thread to make AI move
            self.ui_queue.put(("ai_move", None))
        
        # UI thread loop
        while self.running:
            try:
                # Process any pending UI updates
                if self.fig is None:
                    self.update_board()
                
                # Keep the UI responsive
                plt.pause(0.1)
            except Exception as e:
                print(f"UI thread error: {e}")
                break
                
        # Clean up
        plt.close('all')
        
    def start(self):
        """Start the chess game with proper threading"""
        print("Interactive Chess Game (Threaded Version)")
        print("----------------------------------------")
        print("Enter coordinates to select pieces and make moves.")
        print("Available commands:")
        print("- hint: Get a move suggestion")
        print("- undo: Take back your last move")
        print("- save: Save the current game")
        print("- load: Load a previously saved game")
        print("- resign: Resign the current game")
        print("- cancel: Cancel your current piece selection")
        
        # Start UI thread
        self.ui_thread = threading.Thread(target=self._ui_thread_function)
        self.ui_thread.daemon = True
        self.ui_thread.start()
        
        # Main game loop (in main thread)
        game_over = False
        while not game_over and self.running:
            try:
                # Check for messages from UI thread
                try:
                    message_type, data = self.ui_queue.get(block=False)
                    if message_type == "ai_move":
                        self.make_ai_move()
                        self.ui_queue.task_done()
                except queue.Empty:
                    pass
                
                # Human's turn
                if self.board.turn == self.human_color:
                    move_made = self.process_input()
                    
                    if move_made:
                        # Check if game is over
                        if self.board.is_game_over():
                            game_over = True
                            continue
                        
                        # If not game over, AI makes its move
                        self.make_ai_move()
                        
                        # Check again if game is over
                        if self.board.is_game_over():
                            game_over = True
                
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Game error: {e}")
                break
        
        # Clean up
        self.running = False
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1.0)  # Wait for UI thread to exit
        
        print("Game finished.")