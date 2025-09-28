# backend/uvicorn_start.py

import uvicorn
from app import app # <-- Changed import

if __name__ == "__main__": 
    # Use the app object directly without package path in the string
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)