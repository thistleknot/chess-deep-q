import chess
import matplotlib.pyplot as plt
from game_play import play_game, visual_play_game, visual_play_game_with_features, list_saved_games
from board_utils import print_board, get_move_uci
from chess_ai import OptimizedChessAI
from evaluation import fast_evaluate_position, format_score
from ui import NonClickableChessBoard
#from threaded_board import ThreadedChessBoard
from terminal_board import TerminalChessBoard
import os

# Find the display_full_menu function and modify it to prevent repetition
def display_full_menu(chess_ai):
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
    
    # Show current AHA status in the menu
    aha_status = "ON" if getattr(chess_ai.dqn_agent, 'use_aha_learning', False) else "OFF"
    print(f"14. Toggle AHA Learning (currently: {aha_status})")
    print("15. Configure AHA Learning settings")

    # Show current dynamic-difficulty status in the menu
    dda_status = "ON" if chess_ai.difficulty_settings.get('enabled', False) else "OFF"
    print(f"16. Toggle Dynamic Difficulty (currently: {dda_status})")
    print("17. Configure Dynamic Difficulty settings")
    print("18. Play vs alpha-beta engine (strong, ~1400-1700, CPU)")
    print("19. Play vs learned beam (experimental; net undertrained)")
    print("20. Exit")


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
                terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.BLACK)
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
                    human_color_choice = input("Play as white or black? (w/b, default: w): ").lower() or 'w'
                    human_color = chess.BLACK if human_color_choice.startswith('b') else chess.WHITE
                    
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
            new_state = "enabled" if chess_ai.dqn_agent.use_aha_learning else "disabled"
            print(f"AHA Learning {new_state}")
            
            # If just enabled, show current settings
            if chess_ai.dqn_agent.use_aha_learning:
                print(f"  Budget per game: {chess_ai.dqn_agent.aha_budget_per_game}")
                print(f"  Evaluation threshold: {chess_ai.dqn_agent.aha_threshold}")
                print("  Use option 15 to configure these settings")
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
        elif choice == '16':
            dda = chess_ai.difficulty_settings
            dda['enabled'] = not dda.get('enabled', False)
            print(f"Dynamic Difficulty {'enabled' if dda['enabled'] else 'disabled'}")
            if dda['enabled']:
                print(f"  Regret offset: {dda['offset']}  (+ = stronger opponent, - = handicap)")
                print("  Use option 17 to configure these settings")
        elif choice == '17':
            dda = chess_ai.difficulty_settings
            cal = getattr(chess_ai, 'elo_calibrator', None)
            print("Current Dynamic Difficulty settings:")
            print(f"  Regret offset: {dda['offset']}  (+ = harder, - = easier)")
            if cal is not None:
                status = "calibrated" if cal.is_calibrated() else "NOT calibrated"
                anchor = "measured" if cal.anchor_measured else "assumed"
                print(f"  ELO calibration: {status}; full-strength anchor {cal.anchor_elo:.0f} ({anchor})")
            try:
                dda['offset'] = float(input(f"Enter new regret offset (current: {dda['offset']}): ") or dda['offset'])
                print("Dynamic Difficulty settings updated!")
            except ValueError:
                print("Invalid input. Settings unchanged.")
            # Optional: run/refresh the temperature->ELO calibration (compute-heavy).
            if cal is not None:
                run_cal = input("Run/refresh ELO calibration now? (compute-heavy) (y/n, default n): ").lower() or 'n'
                if run_cal == 'y':
                    from constants import ELO_CALIBRATION_PATH
                    anchor_in = input(f"Full-strength ELO anchor (blank keeps {cal.anchor_elo:.0f}): ").strip()
                    if anchor_in:
                        try:
                            cal.anchor_elo = float(anchor_in)
                            cal.anchor_measured = False
                        except ValueError:
                            print("Invalid anchor; keeping current.")
                    try:
                        games = int(input("Games per temperature (default 20): ") or 20)
                    except ValueError:
                        games = 20
                    print("Calibrating (playing self-play games at several temperatures)...")
                    cal.calibrate(chess_ai, games_per_tau=games)
                    cal.save(ELO_CALIBRATION_PATH)
                    print(f"Calibration saved to {ELO_CALIBRATION_PATH}.")
        elif choice == '18':
            # The strong opponent: pure-Python alpha-beta engine (engine.py), CPU-only.
            from play_engine import main as play_engine_main
            play_engine_main()
        elif choice == '19':
            _play_beam(chess_ai)
        elif choice == '20':
            print("Exiting menu.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 20.")
    
    # Return the updated feature flags
    return verbose, use_visual_board, use_enhanced_features


def _setup_difficulty(chess_ai):
    """Ask how the opponent should scale: full strength, auto-match, or a fixed temperature."""
    dda = chess_ai.difficulty_settings
    cal = getattr(chess_ai, 'elo_calibrator', None)
    print("\nDifficulty:")
    print("  1. Full strength (default)")
    print("  2. Auto-adjust to my skill")
    print("  3. Fixed strength (I set the temperature)")
    choice = input("Choose (1-3, default 1): ").strip() or '1'
    if choice == '2':
        dda['enabled'], dda['mode'] = True, 'auto'
        val = input(f"Harder/easier offset (+ harder / - easier, default {dda['offset']}): ").strip()
        if val:
            try:
                dda['offset'] = float(val)
            except ValueError:
                print("  invalid; keeping default")
        print("Auto-adjust ON: the opponent tracks your level; estimated ELO is shown each turn.")
        if getattr(cal, 'approximate', False):
            print("  (ELO uses an approximate curve — no warm-up needed. Refine it anytime from "
                  "the settings menu.)")
    elif choice == '3':
        dda['enabled'], dda['mode'] = True, 'fixed'
        label = "approx" if getattr(cal, 'approximate', False) else "calibrated"
        print(f"Temperature -> ELO ({label}):")
        for t in sorted(cal.table):
            print(f"    temp {t:>4.2f}  ~{cal.policy_elo(t):.0f} ELO")
        val = input("Set opponent temperature (0 = strongest, higher = weaker; default 0.5): ").strip()
        try:
            dda['fixed_temperature'] = float(val) if val else 0.5
        except ValueError:
            dda['fixed_temperature'] = 0.5
        print(f"  -> opponent is approximately {cal.policy_elo(dda['fixed_temperature']):.0f} ELO ({label})")
        if getattr(cal, 'approximate', False):
            print("  (Approximate curve — no warm-up needed. Refine it anytime from the settings menu.)")
    else:
        dda['enabled'], dda['mode'] = False, 'off'
        print("Full strength.")


def _play_beam(chess_ai):
    """Human vs the experimental net-guided beam (residual tower). Difficulty is the SEARCH BUDGET
    (depth / total_calls), not net temperature — the beam commits argmax. The net is undertrained,
    so expect weak play; this is the architecture path, the strong opponent is menu option 18."""
    print("\nExperimental beam opponent (residual tower; net undertrained -> expect weak play).")
    levels = {"1": (4, 60), "2": (6, 140), "3": (8, 300)}
    print("  1. depth 4 (~60 calls)   2. depth 6 (~140)   3. depth 8 (~300, slower)")
    lvl = input("Search budget (1-3, default 2): ").strip() or "2"
    depth, ops = levels.get(lvl, levels["2"])
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"
    board = chess.Board()
    while not board.is_game_over():
        print("\n" + str(board))
        if (board.turn == chess.WHITE) == human_white:
            raw = input("Your move (SAN/UCI, e.g. Nf3 / g1f3, 'quit'): ").strip()
            if raw in ("quit", "q"):
                return
            try:
                mv = board.parse_san(raw)
            except ValueError:
                try:
                    mv = chess.Move.from_uci(raw)
                    if mv not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Illegal / unparseable move; try again.")
                    continue
            board.push(mv)
        else:
            print("Beam thinking...")
            mv = chess_ai.get_beam_move(board, depth, ops)
            if mv is None:
                return          # tower.pth missing; message already printed
            print(f"Beam plays: {board.san(mv)}")
            board.push(mv)
    print("\n" + str(board))
    print("Result:", board.result())


def main():
    print("Chess AI: value-net TD critic + Russian Doll MCTS (AlphaZero-lite)")
    print("------------------------------------------------------------")
    
    # Create the AI with AHA learning DISABLED by default
    chess_ai = OptimizedChessAI(training_games=20, verbose=True, use_aha_learning=False)  # Changed from True to False

    # Load a saved temperature->ELO calibration if present, so difficulty can show ELO.
    import os
    from elo_calibration import TemperatureEloCalibrator
    from constants import ELO_CALIBRATION_PATH
    if os.path.exists(ELO_CALIBRATION_PATH):
        try:
            chess_ai.elo_calibrator = TemperatureEloCalibrator.load(ELO_CALIBRATION_PATH)
            print(f"Loaded ELO calibration (full-strength anchor "
                  f"{'measured' if chess_ai.elo_calibrator.anchor_measured else 'assumed'} "
                  f"{chess_ai.elo_calibrator.anchor_elo:.0f}).")
        except Exception:
            chess_ai.elo_calibrator = TemperatureEloCalibrator()
    else:
        chess_ai.elo_calibrator = TemperatureEloCalibrator()
    
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
        human_color = chess.BLACK if color_choice.startswith('b') else chess.WHITE

        # Difficulty: full strength, auto-adjust to your skill, or a fixed temperature you set.
        _setup_difficulty(chess_ai)

        # Start the game
        if use_enhanced_features and use_visual_board:
            terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=human_color)
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
            # Check for default model first
            default_model_path = os.path.join('models', 'chess_model.pth')
            if os.path.exists(default_model_path):
                use_default = input(f"Found model at {default_model_path}. Use this? (y/n, default: y): ").lower() or 'y'
                if use_default == 'y':
                    filename = default_model_path
                else:
                    filename = input("Enter model filename: ")
            else:
                filename = input("Enter model filename (will check models/ directory): ")
                # If user didn't specify a path, check models directory
                if filename and not os.path.dirname(filename) and not os.path.exists(filename):
                    models_path = os.path.join('models', filename)
                    if os.path.exists(models_path):
                        filename = models_path
            
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
        
        # Ask about AHA learning
        use_aha = input("Enable AHA Learning (AI learns from mistakes in real-time)? (y/n, default: n): ").lower() or 'n'
        if use_aha == 'y':
            chess_ai.dqn_agent.use_aha_learning = True
            print("AHA Learning enabled - AI will correct mistakes during training")
            
            # Optionally configure AHA settings
            configure_aha = input("Configure AHA settings? (y/n, default: n): ").lower() or 'n'
            if configure_aha == 'y':
                try:
                    budget = int(input(f"AHA budget per game (default: {chess_ai.dqn_agent.aha_budget_per_game}): ") or chess_ai.dqn_agent.aha_budget_per_game)
                    threshold = float(input(f"AHA threshold (default: {chess_ai.dqn_agent.aha_threshold}): ") or chess_ai.dqn_agent.aha_threshold)
                    chess_ai.dqn_agent.aha_budget_per_game = budget
                    chess_ai.dqn_agent.aha_threshold = threshold
                    print(f"AHA settings updated: budget={budget}, threshold={threshold}")
                except ValueError:
                    print("Invalid input. Using default AHA settings.")
        else:
            chess_ai.dqn_agent.use_aha_learning = False
            print("AHA Learning disabled - using standard training")
        
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
            human_color = chess.BLACK if color_choice.startswith('b') else chess.WHITE
            
            if use_enhanced_features and use_visual_board:
                terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=human_color)
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