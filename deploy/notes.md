# Prerequisites

- Linux, python (3.12)
- JupyterLab (or jupyterhub), ipykernel
- mlflow

# Deployment

### Checkout source code i2g-agentnexus

```
# git clone ...
# cd i2g-agentnexus
```

### Install jupyterlab
```
pip install jupyterlab ipykernel 
pip install jupyter-kernel/dist/agnkernel-0.1.0-py3-none-any.whl
```

### install VSP/AI kernel
```
jupyter kernelspec install ./jupyter-kernel/agnkernel/ --user
```

### install dependencies

```
pip install -r requirements.txt
```

### Add openAI API key
```
echo 'OPENAI_API_KEY=<YOUR_KEY>' > .env
```

# Run

### Run mlflow
```
mlflow ui --host 0.0.0.0
```

### Run jupyterlab
```
JUPYTER_TOKEN=12345678 jupyter lab --ip 0.0.0.0 --port 8989
```

### Run Agent service
```
python app.py --agent
```

### Run Tool service
```
python app.py --mcp
```

# Testing

Open browser at `http://<host/ip>:8989`, login with `$JUPYTER_TOKEN`. You should see a kernel named `VSP AI/ML Agent`. Start a console / notebook with this kernel and start chatting

# Final comments

1. Current status:

- This application does not support multiple users. It basically allows single-user only. Multiple users can access and work concurrently but their works will interfere each other.

2. Future:

- Customize UI (Jupyter lab theme and customization)
- Authentication support via jupyterhub
- Custom authentication via LDAP/AD
- Multi-user support (Store, Accepted result, ...)
