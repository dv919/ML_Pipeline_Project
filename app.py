import os
import sys
import traceback

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, render_template

# Templates folder in the root
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('index.html')
        else:
            # Import here to avoid issues at startup
            from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline
            
            # Get data from the form
            data = CustomData(
                year=int(request.form.get('year')),
                month=int(request.form.get('month')),
                region=request.form.get('region'),
                model=request.form.get('model'),
                units_sold=int(request.form.get('units_sold')),
                avg_price_eur=float(request.form.get('avg_price_eur')),
                revenue_eur=float(request.form.get('revenue_eur')),
                bev_share=float(request.form.get('bev_share')),
                premium_share=float(request.form.get('premium_share')),
                gdp_growth=float(request.form.get('gdp_growth'))
            )

            # Convert to DataFrame and predict
            df = data.get_data_as_data_frame()
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(df)

            return render_template('index.html', prediction=round(float(results[0]), 4))
    except Exception as e:
        app.logger.error(f"Error in predict_datapoint: {str(e)}", exc_info=True)
        return render_template('index.html', error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
