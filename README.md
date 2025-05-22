# Chess AI with Deep Q-Learning & Russian Doll MCTS 🏰

*A chess AI that learns from its mistakes and thinks in narrowing circles*

## 🎯 Quick Start

```bash
python main.py
```

## 🖼️ Interface Demo

### Enhanced Terminal Chess Interface

| Version 1.0 | Version 1.1 |
|--------------|-------------|
| ![Chess v1.0](images/chess-v1.0.png) | ![Chess v1.1](images/chess-v1.1.png) |

**Key Visual Features:**
- 🟡 **Selected Piece** - Yellow highlighting
- 🟢 **Possible Moves** - Green squares  
- 🟣 **Secondary Moves** - Magenta (2-move sequences)
- 🔵 **Last Move** - Blue highlighting
- 🔴 **Threatened Squares** - Red background
- 🟦 **Guarded Squares** - Cyan background
- 🟨 **Contested Squares** - Yellow (both threatened + guarded)

## ✨ What Makes This Different

### 🧠 AHA Learning - Learning from Mistakes in Real Time
Your AI doesn't just make moves and hope for the best. When it detects a significant evaluation drop (configurable threshold), it can:
- Immediately update its neural network with the mistake
- Search for a better alternative 
- Self-correct during training (limited budget per game)

### 🎯 Russian Doll MCTS - Smart Search Narrowing
Instead of exploring moves randomly, the search progressively narrows:
```
21 promising moves → 13 best → 8 better → 5 good → 3 solid → 2 strong → 1 choice
```
Each level is weighted by tactical significance (captures, checks, threats, development).

### 🖼️ Enhanced Terminal Interface
- **Real-time highlighting** as you type coordinates
- **Color-coded threats** (red) and protection (cyan)
- **Move possibilities** shown instantly when selecting pieces
- **Score tracking** with live evaluation updates
- **Secondary moves** highlighting 2-move sequences

## 🏗️ Architecture

```
Russian Doll MCTS + Deep Q-Network
         ↓
Weighted Move Sampling by Chess Logic
         ↓  
Parallel Tree Search (Multi-core)
         ↓
CNN Position Evaluation + Game Experience
```

### Core Components
- **chess_ai.py**: Main AI orchestration with training loops
- **neural_network.py**: CNN-based Q-network with AHA learning
- **mcts.py**: Russian Doll MCTS with progressive narrowing
- **evaluation.py**: Chess position scoring (material, mobility, safety, structure)
- **terminal_board.py**: Rich terminal interface with real-time highlighting

## 🎮 Features

### Training
- **Self-play learning** with experience replay
- **Real-time plotting** of evaluation during games
- **Continuing training** from saved models
- **AHA Learning** for mistake correction (configurable)
- **Progress tracking** with loss curves and game statistics

### Playing
- **Interactive terminal** with coordinate input and visual feedback
- **Move hints** from the AI's current policy
- **Undo system** for taking back moves
- **Save/load games** in PGN format
- **Real-time evaluation** display

### Analysis
- **Training progress plots** showing learning curves
- **Game evaluation tracking** across all training games
- **ELO estimation** by playing against Stockfish
- **Performance metrics** and statistical analysis

## 🔧 Technical Details

### Search Algorithm
- **Russian Doll MCTS**: 7 levels of progressive narrowing
- **Weighted sampling**: Moves categorized by tactical importance
- **Parallel processing**: Multi-threaded search using available CPU cores
- **Annealing**: Search parameters adjust based on training progress

### Neural Network
- **CNN Architecture**: 12-channel input (piece positions) → Conv2D layers → Value output
- **Deep Q-Learning**: Experience replay with target network updates
- **CUDA Support**: Automatic GPU acceleration when available
- **AHA Learning**: Real-time mistake correction during training

### Position Evaluation
Comprehensive scoring based on:
- **Material balance** (piece values)
- **Mobility** (legal moves and attacked squares)
- **King safety** (attacks, castling, central exposure)
- **Pawn structure** (doubled, isolated, chains)
- **Space control** (center and extended center)
- **Piece coordination** (defended pieces, development)

## 🎯 Usage Examples

### Basic Training
```python
chess_ai = OptimizedChessAI(training_games=20, verbose=True)
chess_ai.train()
chess_ai.save_model("my_model.pth")
```

### Enable AHA Learning
```python
chess_ai = OptimizedChessAI(
    training_games=50, 
    use_aha_learning=True
)
chess_ai.train()
```

### Continue Training
```python
chess_ai.load_model("existing_model.pth", continue_training=True)
chess_ai.training_games += 30  # Train 30 more games
chess_ai.train()
```

## 🎮 Playing the Game

### Terminal Commands
```bash
e2      # Select piece at e2 (highlights possible moves)
e4      # Move to e4
e2e4    # Complete move notation
hint    # Get AI's suggested move
undo    # Take back your last move
save    # Save current game
resign  # Resign the game
```

### Visual Interface
- 🟡 **Selected piece** highlighted in yellow
- 🟢 **Possible moves** in green
- 🟣 **Secondary moves** (2-move sequences) in magenta  
- 🔵 **Last move** highlighted in blue
- 🔴 **Threatened squares** in red
- 🟦 **Guarded squares** in cyan

## 📊 Training Features

### Real-time Monitoring
- **Live evaluation plots** during each game
- **Score progression** for both players
- **Training metrics** (loss, epsilon, game length)
- **Final position analysis** across all games

### Analysis Tools
- **Performance curves** showing learning progress
- **Material exchange tracking** 
- **Game outcome statistics**
- **ELO rating estimation** via Stockfish play

## 🛠️ Requirements

```bash
pip install torch numpy chess matplotlib colorama tqdm
```

### Optional
- **Stockfish** chess engine (for ELO evaluation)
- **CUDA** compatible GPU (for faster training)

## 🚀 Innovation Highlights

### AHA Learning System
A novel approach where the AI can recognize mistakes during training:
```
Move → Evaluation Drop Detected → Neural Network Update → Better Alternative → Improved Policy
```

### Russian Doll MCTS  
Efficient tree search that concentrates computation on promising moves:
```
Categorical Move Weighting → Progressive Sampling → Narrowing Focus → Best Move Selection
```

### Enhanced User Experience
Real-time visual feedback system that makes chess analysis intuitive and educational.

## 🔮 Future Features & Development Roadmap

### Strategic Evaluation Enhancements

**1. Position-Adaptive Evaluation Weights**
- Implement dynamic evaluation weights that adjust based on position characteristics (e.g., open/closed position, material imbalance)
- Add a position classifier that can recognize key position types and modify evaluation parameters accordingly

**2. Strategic Pattern Recognition**
- Implement detection of common chess patterns (e.g., bishops of opposite colors, IQP positions, minority attacks)
- Add evaluation terms for important strategic themes (e.g., piece quality, color complex weaknesses, compensation for material)
- Create a database of strategic patterns with corresponding evaluation adjustments

**3. Phase-Specific Evaluation**
- Develop separate evaluation functions for opening, middlegame, and endgame
- Implement phase detection to smoothly transition between evaluation functions
- Apply different piece values and positional weights based on game phase
- Integrate specialized endgame evaluations (e.g., king activity becomes more important)

### Search and Learning Improvements

**4. Enhanced MCTS Implementation**
- Replace Python threads with multiprocessing for true parallel search
- Implement batched MCTS with vectorized operations for GPU acceleration
- Add progressive move widening to dynamically adjust search breadth based on position complexity
- Implement threat detection and extension for horizon effect mitigation

**5. Advanced Reinforcement Learning**
- Replace basic Q-learning with TD(λ) or other temporal difference methods with eligibility traces
- Implement prioritized experience replay with importance sampling
- Add curriculum learning progression (start with endgames, advance to middlegames and openings)
- Develop a more nuanced reward function that captures practical winning chances

**6. Hybrid Learning Approach**
- Add supervised learning pre-training using a database of master games
- Implement self-play with expert iteration to combine MCTS and neural guidance
- Create a validation system using classic chess puzzles to test tactical understanding

### Technical Optimizations

**7. Neural Network Architecture Refactoring**
- Replace the current CNN with a more modern architecture using residual connections
- Reduce fully-connected layer size using 1×1 convolutions to maintain spatial information
- Add batch normalization for faster training convergence
- Implement optional model quantization for inference acceleration

**8. Memory Management Enhancements**
- Replace dictionary cache with a proper LRU cache implementation
- Implement Zobrist hashing for efficient position representation
- Add prefetch functionality for likely next positions
- Implement incremental evaluation updates to avoid full recalculation

**9. Optimized Batch Processing**
- Standardize batch processing throughout the system
- Reduce device transfer overhead with larger, less frequent transfers
- Create a dedicated GPU evaluation pipeline for position batches
- Add adaptive batch sizing based on hardware capabilities

**10. Performance Profiling Framework**
- Add comprehensive performance profiling capabilities
- Implement logging of training metrics, search statistics, and system resource usage
- Create a benchmarking system for comparing versions and configurations
- Add A/B testing capability for evaluating system changes

### User Experience Enhancements

**11. Improved Game Analysis**
- Add visualization of the AI's "thought process" during move selection
- Implement move explanation that describes strategic and tactical considerations
- Create position evaluation breakdowns showing contribution of different factors
- Add a feature to analyze user games with commentaries

**12. Training Management Interface**
- Develop a dashboard for monitoring training progress with key metrics
- Add checkpointing and resumable training with configurable parameters
- Implement automatic hyperparameter tuning
- Create visualization of neural network learning progress

**13. Adaptable Playing Strength**
- Add configurable playing strength levels that go beyond simple search depth adjustment
- Implement personality profiles with different strategic preferences
- Create a progressive learning mode that adapts to the user's skill level
- Add coaching features that suggest alternative moves and explain mistakes

### Implementation Priorities

**🔥 High Priority:**
- Position-adaptive evaluation weights (#1)
- Phase-specific evaluation (#3)
- Enhanced MCTS implementation (#4)
- Neural network architecture refactoring (#7)
- Memory management enhancements (#8)

**⚡ Medium Priority:**
- Strategic pattern recognition (#2)
- Advanced reinforcement learning (#5)
- Optimized batch processing (#9)
- Performance profiling framework (#10)
- Improved game analysis (#11)

**📋 Lower Priority:**
- Hybrid learning approach (#6)
- Training management interface (#12)
- Adaptable playing strength (#13)

*These requirements aim to significantly improve both the chess understanding and computational efficiency of the system while maintaining a balance between immediate practical improvements and longer-term architectural enhancements.*

## 📄 License

MIT License - Built for chess enthusiasts and AI researchers.

---

**"In chess, as in learning, the best move often comes after recognizing the worst one."**