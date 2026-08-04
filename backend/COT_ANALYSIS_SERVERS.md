# CoT Analysis Separate Servers Architecture

## Overview

The CoT (Chain-of-Thought) analysis system now runs each analysis in a **separate isolated server instance**. This prevents any analysis from blocking the main server, ensuring all API requests remain responsive.

## Architecture

### Components

1. **Main Server** (`backend/app/main.py`)
   - Handles all UI and API requests
   - Routes CoT analysis requests to separate servers
   - Manages server lifecycle and cleanup

2. **CoT Analysis Server** (`backend/app/cot_analysis_server.py`)
   - Standalone FastAPI server that runs in isolation
   - Handles a single CoT analysis job
   - Automatically shuts down after completion

3. **Server Manager** (`backend/app/server_manager.py`)
   - Spawns and manages separate server instances
   - Handles port allocation (starts from port 9000)
   - Tracks running servers and cleans up completed ones

## How It Works

### Starting an Analysis

1. User requests CoT analysis via `/jobs/{job_id}/cot-analysis`
2. Main server creates a queue entry in `job_db`
3. Server manager spawns a new isolated server process on an available port
4. The new server receives the analysis request and starts processing
5. Main server returns immediately (non-blocking)

### During Analysis

- The analysis server runs completely independently
- Progress updates are written to shared `job_db.json`
- Main server can check status via `/cot-analyses/queue/{cot_job_id}`
- Other API requests to main server are never blocked

### After Completion

- Analysis server saves results to organized folder structure
- Server automatically shuts down after completion
- Background cleanup task (runs every 30 seconds) removes server entries
- Results are accessible via the main server's API

## Benefits

✅ **Complete Isolation**: Each analysis runs in its own process  
✅ **No Blocking**: Main server always responsive  
✅ **Automatic Cleanup**: Servers shut down after completion  
✅ **Resource Management**: Port allocation and process tracking  
✅ **Shared State**: Results accessible via main server API  

## File Structure

```
backend/
├── app/
│   ├── main.py                    # Main server (handles UI/API)
│   ├── cot_analysis_server.py     # Standalone analysis server
│   ├── server_manager.py          # Server lifecycle management
│   └── cot_queue.py               # Queue management (still used)
└── logs/
    └── cot_servers/               # Logs for each analysis server
```

## Port Allocation

- Analysis servers start from port **9000**
- Each new server gets the next available port
- Ports are automatically freed when servers shut down

## Monitoring

- Server logs: `backend/logs/cot_servers/cot_server_{cot_job_id}_{port}.log`
- Main server logs: `backend/logs/cot_analysis_api_{date}.log`
- Status checking: `/cot-analyses/queue/{cot_job_id}`

## Manual Testing

To test a standalone analysis server:

```bash
cd backend
python3 -m app.cot_analysis_server --port 9001
```

Then in another terminal:

```bash
curl -X POST http://127.0.0.1:9001/run-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "cot_job_id": "test_123",
    "job_id": "parent_job_456",
    "config": {"judge_mode": "ALWAYS", "diagnostic": false}
  }'
```

## Troubleshooting

### Server fails to start
- Check port availability: `netstat -tuln | grep 9000`
- Check logs: `backend/logs/cot_servers/cot_server_*.log`

### Analysis not completing
- Check server process: `ps aux | grep cot_analysis_server`
- Verify job_db.json has correct status
- Check main server logs for errors

### Port conflicts
- Increase `base_port` in `server_manager.py`
- Or manually kill stuck processes: `pkill -f cot_analysis_server`


