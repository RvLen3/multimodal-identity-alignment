# Backend API (Scaffold)

This backend is a placeholder service for frontend integration.

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /api/health`
- `POST /api/search`
- `POST /api/verify`
- `GET /api/system/overview`
- `GET /api/tasks/{task_id}`

Current responses are mock data with stable schemas, so frontend can be wired first.
