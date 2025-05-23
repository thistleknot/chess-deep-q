import chess
import matplotlib.pyplot as plt
from game_play import play_game, visual_play_game, visual_play_game_with_features, list_saved_games
from board_utils import print_board, get_move_uci
from chess_ai import OptimizedChessAI
from evaluation import fast_evaluate_position, format_score
from ui import NonClickableChessBoard
#from threaded_board import ThreadedChessBoard
from terminal_board import TerminalChessBoard

# Find the display_full_menu function and modify it to prevent repetition
def display_full_menu(chess_ai):
    """Display the full menu of options after the initial workflow"""
    print("\nChess AI Menu:")
    """Display the full menu of options after the initial workflow"""
    print("\nChess AI Menu:")
    print("1. Play against AI (human as white)")
    print("2. Play against AI (human as black)")
    print("3. Watch AI play against itself")
    print("4. Train more")
    print("5. Continue training from current state")
    print("6. Save model")
    print("7. Load model")
    print("8. Set up custom position (FEN)")
    print("9. Toggle verbose output")
    print("10. Toggle visual board")
    print("11. Toggle enhanced features")
    print("12. Plot final game scores")
    print("13. Evaluate ELO rating")
    print("14. Toggle AHA Learning")
    print("15. Configure AHA Learning settings") 
    print("16. Exit")
    # Removed the infinite while loop that was here


# Update this function in the main code
def handle_menu_selections(chess_ai, verbose, use_visual_board, use_enhanced_features):
    """Handle menu selections and return updated feature flags"""

    # Main menu
    while True:
        display_full_menu(chess_ai)
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            if use_enhanced_features and use_visual_board:
                # Use the non-clickable board with enhanced features
                terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
                terminal_board.start()
            elif use_visual_board:
                visual_play_game(chess_ai, human_color=chess.WHITE)
            else:
                play_game(chess_ai, human_color=chess.WHITE)
        elif choice == '2':
            if use_enhanced_features and use_visual_board:
                # Use the non-clickable board with enhanced features
                threaded_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.BLACK)
                terminal_board.start()
            elif use_visual_board:
                visual_play_game(chess_ai, human_color=chess.BLACK)
            else:
                play_game(chess_ai, human_color=chess.BLACK)
        elif choice == '3':
            # Watch AI play against itself
            ai1 = OptimizedChessAI(verbose=verbose)
            ai2 = OptimizedChessAI(verbose=verbose)
            
            # Load the same model for both if available
            try:
                ai1.load_model("chess_model.pth")
                ai2.load_model("chess_model.pth")
            except FileNotFoundError:
                print("Model file not found. Using untrained models.")
            
            # Create a single shared board
            shared_board = chess.Board()
            move_count = 0
            
            while not shared_board.is_game_over() and move_count < 200:
                print_board(shared_board)
                
                current_ai = ai1 if shared_board.turn == chess.WHITE else ai2
                ai_name = "White AI" if shared_board.turn == chess.WHITE else "Black AI"
                
                print(f"{ai_name} is thinking...")
                
                # Temporarily set the AI's board to the shared board
                current_ai.board = shared_board.copy()  # Use a copy to avoid reference issues
                move = current_ai.get_best_move()
                
                # Validate the move is legal before making it
                if move in shared_board.legal_moves:
                    shared_board.push(move)
                    move_count += 1
                    print(f"{ai_name} plays: {get_move_uci(move)}")
                else:
                    print(f"Error: {ai_name} attempted illegal move {get_move_uci(move)}. Stopping game.")
                    break
            
            print_board(shared_board)
            print(f"Game over. Result: {shared_board.result()}")
        # Other menu options remain the same
        elif choice == '4':
            # Fresh training
            num_games = int(input("Enter number of training games: "))
            chess_ai.training_games = num_games
            progress_interval = max(1, min(num_games // 10, 10))
            chess_ai.train(progress_interval=progress_interval)
            chess_ai.plot_training_progress()
        elif choice == '5':
            # Continue training from current state
            additional_games = int(input("Enter number of additional training games: "))
            original_games = chess_ai.training_games
            chess_ai.training_games += additional_games
            progress_interval = max(1, min(additional_games // 10, 10))
            
            print(f"Continuing training from game {len(chess_ai.game_history)+1} to {chess_ai.training_games}")
            chess_ai.train(progress_interval=progress_interval)
            chess_ai.plot_training_progress()
        elif choice == '6':
            filename = input("Enter filename to save model (default: chess_model.pth): ") or "chess_model.pth"
            chess_ai.save_model(filename)
        elif choice == '7':
            filename = input("Enter filename to load model (default: chess_model.pth): ") or "chess_model.pth"
            continue_training = input("Load training state as well? (y/n): ").lower() == 'y'
            try:
                chess_ai.load_model(filename, continue_training=continue_training)
                print("Model loaded successfully!")
            except FileNotFoundError:
                print(f"Model file {filename} not found.")
        elif choice == '8':
            fen = input("Enter FEN position (e.g., 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'): ")
            try:
                chess_ai.set_board_from_fen(fen)
                print("Position set successfully!")
                print_board(chess_ai.board)
                
                # Ask if user wants to play from this position
                play_choice = input("Do you want to play from this position? (y/n): ").lower()
                if play_choice == 'y':
                    human_color_choice = input("Play as white or black? (w/b): ").lower()
                    human_color = chess.WHITE if human_color_choice == 'w' else chess.BLACK
                    
                    if use_enhanced_features and use_visual_board:
                        terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
                        terminal_board.start()
                    elif use_visual_board:
                        visual_play_game(chess_ai, human_color=human_color)
                    else:
                        play_game(chess_ai, human_color=human_color)
                
            except ValueError:
                print("Invalid FEN position. Please try again.")
        elif choice == '9':
            verbose = not verbose
            chess_ai.verbose = verbose
            print(f"Verbose output {'enabled' if verbose else 'disabled'}")
        elif choice == '10':
            use_visual_board = not use_visual_board
            print(f"Visual board {'enabled' if use_visual_board else 'disabled'}")
        elif choice == '11':
            use_enhanced_features = not use_enhanced_features
            print(f"Enhanced features {'enabled' if use_enhanced_features else 'disabled'}")
            if use_enhanced_features and not use_visual_board:
                use_visual_board = True
                print("Visual board automatically enabled to support enhanced features.")
        elif choice == '12':
            chess_ai.plot_final_game_scores()
        elif choice == '13':
            print("\nELO Rating Evaluation")
            print("--------------------")
            print("This will play your model against Stockfish at various strength levels")
            print("to determine an approximate ELO rating.")
            print("Note: You need Stockfish installed on your system.")
            
            games_per_level = int(input("Number of games per ELO level (default: 20): ") or 20)
            starting_elo = int(input("Starting ELO to test (default: 1500): ") or 1500)
            
            estimated_elo = chess_ai.evaluate_elo_rating(
                num_games=games_per_level,
                starting_elo=starting_elo
            )
            
            if estimated_elo:
                print(f"\nYour model's estimated ELO rating: {estimated_elo}")
        # In handle_menu_selections:
        elif choice == '14':
            current_state = getattr(chess_ai.dqn_agent, 'use_aha_learning', False)
            chess_ai.dqn_agent.use_aha_learning = not current_state
            print(f"AHA Learning {'enabled' if chess_ai.dqn_agent.use_aha_learning else 'disabled'}")

        elif choice == '15':
            print(f"Current AHA settings:")
            print(f"  Budget per game: {chess_ai.dqn_agent.aha_budget_per_game}")
            print(f"  Evaluation threshold: {chess_ai.dqn_agent.aha_threshold}")
            
            try:
                new_budget = int(input(f"Enter new budget per game (current: {chess_ai.dqn_agent.aha_budget_per_game}): ") or chess_ai.dqn_agent.aha_budget_per_game)
                new_threshold = float(input(f"Enter new threshold (current: {chess_ai.dqn_agent.aha_threshold}): ") or chess_ai.dqn_agent.aha_threshold)
                
                chess_ai.dqn_agent.aha_budget_per_game = new_budget
                chess_ai.dqn_agent.aha_threshold = new_threshold
                print("AHA Learning settings updated!")
            except ValueError:
                print("Invalid input. Settings unchanged.")
        else:
            print("Invalid choice. Please enter a number between 1 and 15.")
    
    # Return the updated feature flags
    return verbose, use_visual_board, use_enhanced_features


def main():
    print("Optimized Chess AI with Russian Doll MCTS and Deep Q-Learning")
    print("------------------------------------------------------------")
    
    # Create the AI with default settings
    chess_ai = OptimizedChessAI(training_games=20, verbose=True, use_aha_learning=True)
    
    # First, determine the user's primary goal
    print("\nWhat would you like to do?")
    print("1. Play chess against the AI")
    print("2. Train or improve the AI")
    print("3. Analyze AI performance")
    
    primary_goal = input("Enter your choice (1-3): ")
    
    # Default visual settings
    use_visual_board = True
    use_enhanced_features = True
    verbose = False
    
    # PLAY CHESS PATHWAY
    if primary_goal == '1':
        # Load a pre-trained model by default
        load_model = input("Would you like to load a pre-trained model? (y/n, default: y): ").lower() or 'y'
        if load_model == 'y':
            filename = input("Enter model filename (default: chess_model.pth): ") or "chess_model.pth"
            try:
                chess_ai.load_model(filename)
                print(f"Model loaded from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found. Using untrained model.")
        
        # Visual settings
        visual_board = input("Use visual board for gameplay? (y/n, default: y): ").lower() or 'y'
        use_visual_board = visual_board == 'y'
        
        enhanced_features = 'n'
        if use_visual_board:
            enhanced_features = input("Use enhanced features (score tracking, threat highlighting)? (y/n, default: y): ").lower() or 'y'
        use_enhanced_features = enhanced_features == 'y'
        
        # Choose color
        color_choice = input("Play as white or black? (w/b, default: w): ").lower() or 'w'
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        
        # Start the game
        if use_enhanced_features and use_visual_board:
            terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
            terminal_board.start()
        elif use_visual_board:
            visual_play_game_with_features(chess_ai, human_color=human_color)
        else:
            play_game(chess_ai, human_color=human_color)
    
    # TRAIN AI PATHWAY
    elif primary_goal == '2':
        # Ask about training from existing model
        continue_training = input("Continue training from an existing model? (y/n, default: n): ").lower() or 'n'
        
        if continue_training == 'y':
            filename = input("Enter model filename (default: chess_model.pth): ") or "chess_model.pth"
            try:
                chess_ai.load_model(filename, continue_training=True)
                print(f"Continuing training from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found. Starting with untrained model.")
        
        # Training parameters
        num_games = int(input("Enter number of training games (default: 20): ") or "20")
        chess_ai.training_games = num_games
        
        verbose = input("Enable verbose output during training? (y/n, default: y): ").lower() or 'y'
        chess_ai.verbose = verbose == 'y'
        verbose = chess_ai.verbose  # Update local variable for menu
        
        # Start training
        print(f"\nStarting training for {chess_ai.training_games} games...")
        progress_interval = max(1, min(chess_ai.training_games // 10, 10))
        chess_ai.train(progress_interval=progress_interval)
        
        # Training results are now shown automatically after training, including the final game scores plot
        
        # Save model option
        save_model = input("Save the trained model? (y/n, default: y): ").lower() or 'y'
        if save_model == 'y':
            filename = input("Enter filename (default: chess_model.pth): ") or "chess_model.pth"
            chess_ai.save_model(filename)
            print(f"Model saved to {filename}")
        
        # Play option after training
        play_after_training = input("Would you like to play against the trained AI? (y/n, default: y): ").lower() or 'y'
        if play_after_training == 'y':
            visual_board = input("Use visual board for gameplay? (y/n, default: y): ").lower() or 'y'
            use_visual_board = visual_board == 'y'
            
            enhanced_features = 'n'
            if use_visual_board:
                enhanced_features = input("Use enhanced features? (y/n, default: y): ").lower() or 'y'
            use_enhanced_features = enhanced_features == 'y'
            
            color_choice = input("Play as white or black? (w/b, default: w): ").lower() or 'w'
            human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
            
            if use_enhanced_features and use_visual_board:
                terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
                terminal_board.start()
            elif use_visual_board:
                visual_play_game_with_features(chess_ai, human_color=human_color)
            else:
                play_game(chess_ai, human_color=human_color)
    
    # ANALYZE AI PATHWAY
    elif primary_goal == '3':
        # Load a model for analysis
        load_model = input("Load a specific model for analysis? (y/n, default: y): ").lower() or 'y'
        if load_model == 'y':
            filename = input("Enter model filename (default: chess_model.pth): ") or "chess_model.pth"
            try:
                chess_ai.load_model(filename)
                print(f"Model loaded from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found. Using untrained model.")
        
        # Analysis options
        print("\nAnalysis options:")
        print("1. Watch AI play against itself")
        print("2. Plot training progress")
        print("3. Plot final game scores")
        print("4. Evaluate ELO rating")
        print("5. Return to main menu")
        
        while True:
            analysis_choice = input("Choose an analysis option (1-5): ")
            
            if analysis_choice == '1':
                # Self-play
                print("\nAI vs AI game:")
                ai1 = chess_ai  # Use the loaded model
                ai2 = OptimizedChessAI(verbose=False)
                try:
                    ai2.load_model(filename)  # Use the same model for opponent
                except:
                    print("Using untrained model for opponent")
                
                # Visual options
                visual = input("Use visual board for self-play? (y/n, default: y): ").lower() or 'y'
                show_scores = input("Display and track scores during play? (y/n, default: y): ").lower() or 'y'
                
                # Create a shared board
                shared_board = chess.Board()
                move_count = 0
                
                # For score tracking
                white_scores = []
                black_scores = []
                
                # Setup score plot if requested
                if show_scores and visual == 'y':
                    plt.ion()  # Turn on interactive mode
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title("Game Progress Scores")
                    ax.set_xlabel("Move Number")
                    ax.set_ylabel("Score")
                    ax.grid(True)
                    white_line, = ax.plot([], [], 'b-', label="White")
                    black_line, = ax.plot([], [], 'r-', label="Black")
                    ax.legend()
                    plt.show(block=False)
                
                # Game loop
                while not shared_board.is_game_over() and move_count < 200:
                    if visual == 'y':
                        print_board(shared_board)
                    
                    current_ai = ai1 if shared_board.turn == chess.WHITE else ai2
                    ai_name = "White AI" if shared_board.turn == chess.WHITE else "Black AI"
                    
                    print(f"{ai_name} is thinking...")
                    current_ai.board = shared_board.copy()
                    move = current_ai.get_best_move()
                    
                    if move in shared_board.legal_moves:
                        shared_board.push(move)
                        move_count += 1
                        
                        # Score evaluation and display
                        if show_scores:
                            raw_score = fast_evaluate_position(shared_board)
                            white_scores.append(raw_score)
                            black_scores.append(-raw_score)
                            
                            # Format scores for display
                            white_score_str = format_score(raw_score)
                            black_score_str = format_score(-raw_score)
                            
                            print(f"{ai_name} plays: {get_move_uci(move)} - White: {white_score_str} | Black: {black_score_str}")
                            
                            # Update plot if visual
                            if visual == 'y' and move_count % 3 == 0:  # Update every 3 moves for efficiency
                                move_numbers = list(range(1, len(white_scores) + 1))
                                white_line.set_data(move_numbers, white_scores)
                                black_line.set_data(move_numbers, black_scores)
                                
                                # Adjust plot limits
                                ax.set_xlim(0, max(move_count + 5, 50))
                                ax.set_ylim(min(min(white_scores), min(black_scores)) - 2, 
                                            max(max(white_scores), max(black_scores)) + 2)
                                
                                # Update plot
                                fig.canvas.draw_idle()
                                fig.canvas.flush_events()
                        else:
                            print(f"{ai_name} plays: {get_move_uci(move)}")
                    else:
                        print(f"Error: {ai_name} attempted illegal move. Stopping game.")
                        break
                
                # Final position
                if visual == 'y':
                    print_board(shared_board)
                
                # Display final result
                result = shared_board.result()
                print(f"Game over. Result: {result}")
                
                # Final scores
                if show_scores:
                    final_score = fast_evaluate_position(shared_board)
                    print(f"Final evaluation - White: {format_score(final_score)} | Black: {format_score(-final_score)}")
                    
                    # Plot final score distribution if requested
                    if input("Plot game score progression? (y/n): ").lower() == 'y':
                        plt.figure(figsize=(12, 6))
                        move_nums = list(range(1, len(white_scores) + 1))
                        plt.plot(move_nums, white_scores, 'b-', label="White")
                        plt.plot(move_nums, black_scores, 'r-', label="Black")
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        plt.grid(True, alpha=0.3)
                        plt.title("Score Progression During Game")
                        plt.xlabel("Move Number")
                        plt.ylabel("Evaluation Score")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                
                # Close any open plots
                if show_scores and visual == 'y':
                    plt.ioff()
                    plt.close()
            
            elif analysis_choice == '2':
                # Plot training progress
                if hasattr(chess_ai, 'loss_history') and chess_ai.loss_history:
                    chess_ai.plot_training_progress()
                else:
                    print("No training data available. Train the model first.")
            
            elif analysis_choice == '3':
                # Plot final game scores
                if hasattr(chess_ai, 'game_history') and chess_ai.game_history:
                    chess_ai.plot_final_game_scores()
                else:
                    print("No game history available. Train the model first.")
            
            elif analysis_choice == '4':
                # ELO rating evaluation
                print("\nELO Rating Evaluation")
                print("This will play your model against Stockfish at various strength levels.")
                print("Note: You need Stockfish installed on your system.")
                
                games_per_level = int(input("Number of games per ELO level (default: 10): ") or "10")
                starting_elo = int(input("Starting ELO to test (default: 1500): ") or "1500")
                
                try:
                    estimated_elo = chess_ai.evaluate_elo_rating(
                        num_games=games_per_level,
                        starting_elo=starting_elo
                    )
                    if estimated_elo:
                        print(f"\nYour model's estimated ELO rating: {estimated_elo}")
                except Exception as e:
                    print(f"Error evaluating ELO rating: {e}")
                    print("Make sure Stockfish is installed and accessible.")
            
            elif analysis_choice == '5':
                break
            
            print()  # Add spacing between analyses
    
    # INVALID CHOICE
    else:
        print("Invalid choice. Please restart the program and select a valid option.")
    
    show_menu = input("\nWould you like to see the full menu of options? (y/n, default: n): ").lower() or 'n'
    if show_menu == 'y':
        # Now we call our menu handler which has its own loop
        verbose, use_visual_board, use_enhanced_features = handle_menu_selections(
            chess_ai, verbose, use_visual_board, use_enhanced_features
        )

if __name__ == "__main__":
    main()