# Run the FastAPI server
$ErrorActionPreference = "Stop"
Write-Host "Starting Razorpay AI Risk Manager API Server..." -ForegroundColor Cyan
.\venv\Scripts\python.exe src/api/server.py
