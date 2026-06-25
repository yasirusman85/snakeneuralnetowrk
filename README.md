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

## Run the backend
```bash
uvicorn backend.app.main:app --reload
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
