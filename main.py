"""
Main entry point for the BMW Sales Prediction ML Pipeline Project.

This project provides:
1. Training Pipeline: Process data, train models, and save artifacts
2. Prediction App: Web interface for making predictions

Usage:
    python main.py train    # Run the training pipeline
    python main.py app      # Run the prediction web app
"""

import sys
import argparse

def run_training():
    """Run the complete training pipeline."""
    print("🚀 Starting BMW Sales Prediction Training Pipeline...")
    try:
        from src.pipelines.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        score = pipeline.start_training()
        print(f"✅ Training completed! Best R² Score: {score}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

def run_app():
    """Run the Flask prediction web app."""
    print("🌐 Starting BMW Sales Prediction Web App...")
    try:
        from app import app
        print("✅ App loaded successfully!")
        print("🌐 Open your browser and go to: http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        print(f"❌ App failed to start: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="BMW Sales Prediction ML Pipeline")
    parser.add_argument(
        'command',
        choices=['train', 'app'],
        help='Command to run: train (training pipeline) or app (web app)'
    )

    args = parser.parse_args()

    if args.command == 'train':
        run_training()
    elif args.command == 'app':
        run_app()

if __name__ == "__main__":
    main()