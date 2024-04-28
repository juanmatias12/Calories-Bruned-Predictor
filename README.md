# AI-Powered Caloric Expenditure Prediction

## Project Overview
This AI-driven project predicts the caloric burn based on user activity, leveraging machine learning algorithms to provide accurate and personalized health insights.

## Functionality and Correctness
- **Core Algorithms**: Implements Linear Regression, RandomForest, GBM, and XGBRegressor.
- **Expected Results**: The application predicts caloric expenditure post-exercise.

## Installation
Install dependencies with:
```
pip install -r requirements.txt
```
## Clone the Repository
```
git glone https://github.com/juanmatias12/Calories-Burned
```
## Usage
Train models and evalute their performance with: 
```
python main.py
```

Run the Streamlit application for use:
```
streamlit run application.py
```
## Data and Methodology
* Preprocessing: Clean, normalize, and encode the dataset.
* Algorithms: Employed various regression models for prediction tasks.
* Experimental Results
* Model performance assessed using Mean Absolute Error (MAE), with detailed insights into each model's accuracy.

## Visualization
Visual aids like histograms and scatter plots are used to illustrate the model's predictions versus actual values.

## Discussion
Discusses AI's predictive power, the variability in physical activities, and the robustness to outliers.

## Conclusion and Learning Experience
Reflects on AI's potential in the health and fitness industry, highlighting personal growth and project contributions.

## Real-World Application
Demonstrates a Streamlit app that uses the XGBRegressor for real-world user predictions.

## External Libraries and APIs
This project utilizes several external libraries, essential for the execution of machine learning algorithms and data handling. Below are the libraries used, along with their purpose and links to their official documentation:

* Pandas: Used for data manipulation and analysis. [Pandas Documentation](https://pandas.pydata.org/docs/#module-pandas)
* NumPy: Fundamental package for numerical computation in Python. [NumPy Documentation](https://numpy.org/devdocs/)
* Scikit-learn: Employed for implementing machine learning models, including data splitting and performance metrics. [Scikit-learn Documentation](https://https://scikit-learn.org/stable/)
* XGBoost: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It is used here for the XGBRegressor model. [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
* Matplotlib and Seaborn: These libraries are used for creating visualizations to analyze the data and results. [Matplotlib Documentation](https://matplotlib.org/stable/index.html) and [Seaborn Documentation](https://seaborn.pydata.org)


## Documentation and Code Structure
* README: Details on how to run and understand the project.
* Inline Comments: Clarify the purpose and functionality of code segments.
## Version Control and Collaboration
* Git: Utilized for version control with clear commit messages.

## Additional Materials
* Streamlit Application Demo
* <div>
    <a href="https://www.loom.com/share/56c0ea5333c946cd839a0376d55f97ab">
    </a>
    <a href="https://www.loom.com/share/56c0ea5333c946cd839a0376d55f97ab">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/56c0ea5333c946cd839a0376d55f97ab-with-play.gif">
    </a>
  </div>
* Appendices: 
```
# Example of data preprocces snippet
def preprocess_data(data):
    # Check if the 'Gender' column exists and convert it. Can't use string values. 
    if 'Gender' in data.columns:
        # Map 'male' to 0 and 'female' to 1
        gender_map = {'male': 0, 'female': 1}
        data['Gender'] = data['Gender'].map(gender_map)
```

## Contact
For queries, reach out at jmatias7069@sdsu.edu

## License





