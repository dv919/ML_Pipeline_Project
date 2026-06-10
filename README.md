# BMW Sales Fuel Price Index Predictor

A complete machine learning pipeline for predicting fuel price index based on BMW global sales metrics from 2018-2025. This project demonstrates end-to-end ML development including data ingestion, transformation, model training with regularization, and a web interface for predictions.

## Project Overview

This project predicts the **Fuel Price Index** using BMW sales data as input features. The pipeline includes:
- Robust data preprocessing with outlier detection
- Multiple ML algorithms with hyperparameter optimization
- Cross-validation to detect and prevent overfitting
- A Flask web application for real-time predictions
- Comprehensive logging and error handling

## Features

- **Data Ingestion**: Load and preprocess BMW sales data with train-test split
- **Data Transformation**: Handle outliers using IQR method, feature scaling, and categorical encoding
- **Model Training**: Train multiple ML models with regularization and hyperparameter tuning
- **Overfitting Prevention**: 5-fold cross-validation and regularization parameters
- **Model Evaluation**: Compare model performance with R² scores, RMSE, and MAE
- **Web Application**: Interactive web interface for making predictions
- **Logging**: Comprehensive logging of all pipeline steps

## Project Structure

```
├── data/                          # Raw data files
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and train-test splitting
│   │   ├── data_transformation.py # Outlier removal and feature preprocessing
│   │   └── model_trainer.py       # Model training with regularization
│   ├── pipelines/
│   │   ├── training_pipeline.py   # Complete training workflow
│   │   └── prediction_pipeline.py # Prediction workflow
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions with CV scoring
├── artifacts/                     # Trained models and preprocessors
├── templates/                     # Flask HTML templates
├── logs/                          # Application logs
├── main.py                        # CLI entry point
├── app.py                         # Flask application
├── setup.py                       # Package configuration
└── requirements.txt               # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dv919/ML_Pipeline_Project.git
   cd BMW-Sales-Fuel-Price-Predictor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training pipeline to process data and train models with proper cross-validation:

```bash
python main.py train
```

This will:
- Load BMW sales data (3,072 records from 2018-2025)
- Remove outliers using IQR method (only on training data)
- Preprocess features with scaling and encoding
- Train multiple models with regularization:
  - Random Forest (max_depth=15, min_samples_leaf=5)
  - Decision Tree (max_depth=10 for regularization)
  - Gradient Boosting (early stopping, validation fraction)
  - XGBoost (early stopping rounds)
  - AdaBoost Regressor
  - Linear Regression
- Use 5-fold cross-validation for hyperparameter tuning
- Save the best model and preprocessor to `artifacts/`

### Running the Web App

Start the prediction web application:

```bash
python main.py app
```

Open your browser and navigate to `http://localhost:5000` to make predictions.

### Alternative Commands

```bash
# Run training pipeline directly
python src/pipelines/training_pipeline.py

# Run web app directly
python app.py
```

## Dataset

The model uses BMW global sales data with 3,072 records and 11 features:

**Features:**
- `Year`: 2018-2025
- `Month`: 1-12
- `Region`: Europe, North America, Asia Pacific, etc.
- `Model`: 3 Series, 5 Series, X3, X5, X7, etc.
- `Units_Sold`: Number of vehicles sold
- `Avg_Price_EUR`: Average price in EUR
- `Revenue_EUR`: Total revenue in EUR
- `BEV_Share`: Battery Electric Vehicle market share
- `Premium_Share`: Premium model market share
- `GDP_Growth`: Regional GDP growth percentage

**Target:**
- `Fuel_Price_Index`: Fuel price index to predict (regression task)

## Key Improvements (v0.2)

1. **Overfitting Prevention**:
   - Increased cross-validation folds from 3 to 5
   - Added regularization parameters to tree-based models
   - Early stopping for gradient boosting and XGBoost

2. **Better Model Evaluation**:
   - Added train/test R² comparison to detect overfitting
   - Implemented cross-validation scoring
   - Added RMSE and MAE metrics

3. **Hyperparameter Tuning**:
   - Focused on regularization rather than aggressive parameters
   - Reduced search space for faster training
   - Added validation_fraction and early stopping

4. **Logging and Monitoring**:
   - Detailed logging of model performance metrics
   - Overfitting gap calculation (train R² - test R²)
   - Cross-validation score with confidence intervals

## Model Performance

The pipeline evaluates six regression algorithms and selects the best performer based on cross-validation R² score. Expected performance:
- Training R² typically 0.85-0.95
- Test R² typically 0.75-0.90
- Small overfitting gap indicates good generalization

## Technologies Used

- **Python 3.8+**: Core language
- **Scikit-learn**: ML algorithms and preprocessing
- **XGBoost**: Gradient boosting with early stopping
- **Flask**: Web framework for predictions
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Joblib**: Model and preprocessor serialization
- **Matplotlib/Seaborn**: Data visualization

## Error Handling and Logging

- Custom exception handling for debugging
- Detailed logging to `logs/` directory with timestamps
- Error messages include file names and line numbers
- Application continues with fallback behavior where possible

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with proper testing
4. Run the training pipeline to verify
5. Commit your changes with descriptive messages
6. Push to the branch and submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Model Development Notes

- The model predicts fuel price index, which has low variance across dataset
- Cross-validation helps identify which models generalize best
- Data is well-balanced with no missing values
- Regularization is crucial for preventing overfitting on this relatively small dataset (3,072 records)