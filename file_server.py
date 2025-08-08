from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import secrets


TOKEN = "SECRET123"
route_configs = [
    {
        "path": "/data",
        "file_path": "./data",
        "auth": True,
    },
    {
        "path": "/mlflow",
        "file_path": "./mlartifacts",
        "auth": False,
    },
    {
        "path": "/",
        "file_path": "/tmp",
        "auth": False,
    },
]


app = FastAPI()


def verify_token(token: str):
    if token != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_file_or_404(file_path: Path) -> FileResponse:
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


for cfg in route_configs:
    base_path = Path(cfg["file_path"]).resolve()
    require_auth = cfg.get("auth", False)

    if require_auth:
        async def serve_file(
            file_path: str,
            token: str = Depends(verify_token),
            base_path=base_path
        ):
            return get_file_or_404(base_path / file_path)

    else:
        async def serve_file(file_path: str, base_path=base_path):
            return get_file_or_404(base_path / file_path)
            
    app.get(f"{cfg['path']}{{file_path:path}}")(serve_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("file_server:app", host="0.0.0.0", port=9998)
