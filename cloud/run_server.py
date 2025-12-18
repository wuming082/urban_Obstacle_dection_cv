# cloud/run_server.py
from app.main import app
from app.core.config import get_config
import uvicorn

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload
    )