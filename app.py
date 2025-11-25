# app.py
from train import app

if __name__ == "__main__":
    # Use PORT from environment variable for Render
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
