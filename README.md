# Solar-Prediction
Solar Prediction using XgBoost and using MultiLayer Perceptron for prediction

# Project Overview:

The Solar Prediction project aimed to develop a predictive model for solar energy generation based on various input features. The project utilized machine learning techniques to analyze historical data and make accurate predictions for future solar energy production.

# Project Steps:

  1. Importing Libraries: The project began with importing necessary libraries such as NumPy, Pandas, scikit-learn, and XGBoost to facilitate data manipulation, analysis, and modeling.

  2. Loading Data: Historical solar energy production data was loaded into the project. This dataset served as the foundation for training and evaluating the predictive model.

  3. Data Wrangling: The loaded data underwent preprocessing steps to handle missing values, outliers, and any inconsistencies. Data cleaning and transformation techniques were employed to ensure data quality and reliability.

  4. Feature Selection using Correlation Matrix: A correlation matrix was constructed to identify the relationships between the input features and the target variable (solar energy production). Features with a strong correlation to the target were selected for further analysis.

  5. Feature Selection using SelectKBest Method: The SelectKBest method was applied to rank and select the most informative features for solar energy prediction. This approach helped to reduce dimensionality and focus on the most relevant input variables.

  6. Feature Selection using Extra Tree Classifier: An Extra Tree Classifier algorithm was employed to evaluate the importance of each feature. This technique determined the significance of each input variable in predicting solar energy generation.

  7. Feature Engineering: Several feature engineering techniques such as Box-Cox transformation, logarithmic transformation, Min-Max scaling, and standardization were applied to enhance the predictive power of the selected features. These transformations ensured that the data adhered to certain assumptions and improved the model's performance.

  8. Preparing Data: The dataset was split into training and testing sets. Standardization techniques were applied to normalize the features, ensuring that they had zero mean and unit variance. The data splitting process allowed the model to learn patterns from the training data and evaluate its performance on unseen testing data.

  9. Prediction with XGBoost: The XGBoost algorithm, a popular gradient boosting technique, was employed to build a predictive model for solar energy production. The selected features and preprocessed data were used to train the XGBoost model. This algorithm utilized an ensemble of decision trees to make accurate predictions.

  10. Using MultiLayer Perceptron for Prediction: In addition to XGBoost, a MultiLayer Perceptron (MLP) algorithm was also utilized for solar energy prediction. MLP is a type of artificial neural network that can learn complex relationships between inputs and outputs. It was trained on the preprocessed data to forecast solar energy generation.

# Results:

The project evaluated the performance of the predictive models using various evaluation metrics.

  - XGBoost Results:
    
     - Testing performance:
       
        - Root Mean Square Error (RMSE): 81.44
        - R-squared (R2): 0.93

  - MultiLayer Perceptron Results:
    
     - Mean Absolute Error (MAE): 40.678

These results indicated the accuracy and reliability of the developed models in predicting solar energy generation. The XGBoost model achieved a relatively low RMSE and a high R2 score, suggesting a good fit to the data. The MLP model demonstrated a mean absolute error of 40.678, indicating its ability to predict solar energy production with reasonable accuracy.

In conclusion, the Solar Prediction project successfully developed predictive models using XGBoost and MLP algorithms. The models leveraged feature selection techniques, feature engineering, and data preprocessing to achieve accurate predictions for solar energy generation. The project's findings could be valuable for optimizing solar energy production and contributing to renewable energy planning and management.
