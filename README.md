# Neural Snake

Advanced Snake with a modular Python game core, FastAPI backend, and React frontend.

## Folder structure
- [game/](game) — pygame game, RL agent, environment, renderer, and service layer
- [backend/](backend) — FastAPI API for status, reset, and training control
- [frontend/](frontend) — React + Vite dashboard scaffold
- [shared/](shared) — optional shared utilities and contracts

## Run the game
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the game:
   ```bash
   python -m game.main
   ```

## Game Controls

| Key       | Action                     |
|-----------|----------------------------|
| **T**     | Cycle modes: TRAIN ↔ TEST  |
| **SPACE** | Pause / Resume             |
| **R**     | Reset current episode      |
| **S**     | Save model checkpoint      |
| **L**     | Load model checkpoint      |
| **ESC**   | Quit                       |

## Game Modes

### 🔵 TRAIN Mode
- Agent learns via Deep Q-Learning with epsilon-greedy exploration
- Network weights update every batch (128 steps)
- Epsilon decays from 1.0 to 0.02 over time (controlled exploration)
- Watch the UI panel to see epsilon, reward, and loss metrics
- **Press T** to stop training and evaluate the learned policy

### 🟢 TEST Mode
- Agent plays using **learned policy only** (no exploration, no learning)
- Uses greedy action selection (highest Q-value)
- Network weights are frozen (no updates)
- Useful for evaluating model performance after training
- **Press T** to switch back to training mode

## Typical Workflow

1. **Start training**:
   ```bash
   python -m game.main
   ```
   Game launches in TRAIN mode. Agent explores and learns.

2. **Monitor training**:
   - Watch the on-screen panel for epsilon (exploration rate), loss, and best score
   - Epsilon should slowly decay from 1.0 to ~0.02
   - Best score should gradually improve over time

3. **Evaluate the learned policy** (after ~2-5 minutes of training):
   - Press **T** to enter TEST mode
   - Watch the agent play using only what it learned (no more exploration)
   - Observe if the snake reaches higher scores with the learned greedy policy
   - Press **T** to resume training or **SPACE** to pause

4. **Save progress**:
   - Press **S** at any time to save the model
   - Press **L** to restore a previous checkpoint
   - Model saves to `neural_snake_model.npz`

## Run the backend
From the project root:
```bash
uvicorn backend.app.main:app --reload
```

From inside the backend folder:
```bash
uvicorn main:app --reload
```

## Run the frontend
```bash
cd frontend
npm install
npm run dev
```

## What the frontend can do
- Show live training metrics
- Control training start/stop/reset
- Display checkpoints and scores
- Render charts for reward and epsilon

## Next upgrade path
- Add WebSocket updates from FastAPI to React
- Move model checkpoints to a dedicated storage layer
- Add a proper training scheduler and experiment tracking
