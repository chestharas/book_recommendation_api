import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import check_environment, settings

def main():
    print("=== Book Recommendation API Server ===")
    
    # Check environment
    status = check_environment()
    
    if not status['ready_for_api']:
        print("\n❌ Environment not ready for API!")
        print("Please run training first: python scripts/run_training.py")
        return False
    
    print("\n✅ Environment ready for API!")
    print(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    print(f"Debug mode: {settings.DEBUG}")
    
    try:
        # Run API server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "api.main:app",
            "--host", settings.API_HOST,
            "--port", str(settings.API_PORT)
        ]
        
        if settings.DEBUG:
            cmd.append("--reload")
        
        print(f"Command: {' '.join(cmd)}")
        print("\n--- Starting API Server ---")
        print("📚 API Documentation will be available at:")
        print(f"   • Swagger UI: http://{settings.API_HOST}:{settings.API_PORT}/docs")
        print(f"   • ReDoc: http://{settings.API_HOST}:{settings.API_PORT}/redoc")
        print("\n⏹️  Press Ctrl+C to stop the server")
        
        # Run the server (this will block)
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return False

if __name__ == "__main__":
    main()