# scripts/run_training.py
"""
Script to run the model training
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import check_environment

def main():
    print("=== Book Recommendation Model Training ===")
    
    # Check environment
    status = check_environment()
    
    if not status['ready_for_training']:
        print("\n❌ Environment not ready for training!")
        return False
    
    print("\n✅ Environment ready for training!")
    print("\n--- Starting Training ---")
    
    try:
        # Run training script
        result = subprocess.run([
            sys.executable, 
            str(project_root / 'models' / 'train_model.py')
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\nTraining output:")
            print(result.stdout)
            
            # Check if model files were created
            status_after = check_environment()
            if status_after['ready_for_api']:
                print("\n🚀 Ready to start API server!")
                print("Run: python scripts/run_api.py")
            else:
                print("\n⚠️  Model files may not have been created properly.")
            
            return True
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running training script: {e}")
        return False

if __name__ == "__main__":
    main()