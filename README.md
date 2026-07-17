# Chess RL from scratch — the Trivium Recipe 🏰
*A from-scratch RL agent whose champion holds a **1724 Elo floor** (49W-1D-0L over 50 games
at depth 9 vs Stockfish@1320) — trained on pure self-play with nothing deeper than a
2-ply glance*

## ⭐ The enshrined lesson

**Sparse-depth trivium RL works.** Compound value targets — the *trivium*:
`λ-return : search-value : outcome`, weights **annealed on an Optuna-tuned schedule** — let a
linear eval climb from scratch on pure self-play with **no deep search anywhere in training**:
λ replaces depth, a 2-ply glance keeps targets sound, the outcome term anchors early and
anneals away. Depth is spent only at *play time*, where it converts the learned eval into
strength.

**Current champion (`models/champion.pt`): the amap-897 net** — pst-769 planes ⊕ attack-coverage
maps, the campaign's first confirmed feature win (+51/+72 Elo across two independent seeds).
The 1320 anchor saturates above ~1700, so the honest claim is a floor, not a point estimate.

## 🎯 Quick Start
```bash
pip install -r requirements.txt
python main.py     # play the champion in the terminal
python app.py      # training console (browser UI at http://127.0.0.1:8000/)
```

## 🖼️ In the thick of it

| Queen selected — coverage map live (threatened red, guarded cyan, contested cream) |
|--------------|
| ![Gameplay](images/chess-v1.5-midgame.png) |

Regenerate anytime: `python experiments/board_still.py <seed>`.

## 📚 Documentation

| Doc | What's in it |
|-----|--------------|
| [`docs/GUIDE.md`](docs/GUIDE.md) | Playing commands, training console & Optuna protocol, architecture, release history, roadmap |
| [`spec/trivium.spec.md`](spec/trivium.spec.md) | Canonical spec of the recipe |
| [`docs/LESSONS.md`](docs/LESSONS.md) | Enshrined lessons from the campaign |
| [`docs/ROLLBACK.md`](docs/ROLLBACK.md) | Rollback map |
| [`spec/dispositioned.md`](spec/dispositioned.md) | Everything superseded |
| [`future_exploration.md`](future_exploration.md) | Live research frontier (arm ledger, verdicts) |

## 📄 License
MIT License - Built for chess enthusiasts and AI researchers.

---
**"In chess, as in learning, the best move often comes after recognizing the worst one."**
