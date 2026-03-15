# BMW Sales Prediction ML Pipeline

A complete machine learning pipeline for predicting BMW sales metrics using historical sales data from 2018-2025.

## Features

- **Data Ingestion**: Load and preprocess BMW sales data
- **Data Transformation**: Handle outliers, scaling, and encoding
- **Model Training**: Train multiple ML models with hyperparameter tuning
- **Web Application**: Interactive web interface for predictions
- **Model Evaluation**: Compare model performance with R² scores

## Project Structure

```
├── data/                          # Raw data files
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Feature preprocessing
│   │   ├── model_trainer.py       # Model training and evaluation
│   │   └── app.py                 # Flask web application
│   ├── pipelines/
│   │   ├── training_pipeline.py   # Complete training workflow
│   │   └── prediction_pipeline.py # Prediction workflow
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions
├── artifacts/                     # Trained models and preprocessors
├── templates/                     # Flask HTML templates
├── logs/                          # Application logs
└── requirements.txt               # Python dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training pipeline to process data and train models:

```bash
python main.py train
```

This will:
- Load BMW sales data
- Preprocess features (outlier removal, scaling, encoding)
- Train multiple models (Random Forest, XGBoost, etc.)
- Save the best model and preprocessor to `artifacts/`

### Running the Web App

Start the prediction web application:

```bash
python main.py app
```

Open your browser and go to `http://localhost:5000` to make predictions.

### Alternative Commands

You can also run components individually:

```bash
# Run training pipeline directly
python src/pipelines/training_pipeline.py

# Run web app directly
python src/components/app.py
```

## Data

The model uses BMW global sales data from 2018-2025 with features:
- Year, Month, Region
- Model (3 Series, 5 Series, X3, X5, etc.)
- Units Sold, Average Price, Revenue
- BEV Share, Premium Share, GDP Growth
- **Target**: Fuel Price Index

## Model Performance

The pipeline evaluates multiple algorithms and selects the best performer based on R² score. Current best model achieves ~99.8% accuracy.

## Technologies Used

- **Python** - Core language
- **Scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting
- **Flask** - Web framework
- **Pandas/Numpy** - Data manipulation
- **Joblib** - Model serialization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License.