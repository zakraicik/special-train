# Special Train

`special-train` contains the code needed to train an Ethereum price forecasting model locally. The repository is organized into three major sections. **Please note:** You will need your own AWS and Polygon API keys stored in the AWS Secrets Manager using the same keys to execute these scripts.

The following graph shows the predicted vs. actual Ethereum prices. The tan line is the actual ethereum price, and the red line is our prediction. I hid the legend because it's a multistep prediction, so it get's very crowded.

![Predicted vs. Actual Ethereum Prices](https://github.com/zakraicik/special-train/blob/main/special_train/eval/actual_vs_expected.png)

## Repository Structure

1. **`data`**: Contains scripts required to download and process data.

   - **`get_data.py`**: Retrieves raw Ethereum price data at 5-minute intervals. The data is stored in an S3 bucket. This script is executed weekly to collect new data using GitHub Actions.
   - **`preprocess.py`**: Prepares the data for use in an LSTM model. This script creates training, validation, and test sets, scales these sets, and generates input sequences for model training.

2. **`train`**: Contains scripts to train the model.

   - **`train.py`**: Uses Keras Tuner to create an LSTM model that minimizes validation loss (Mean Absolute Error, MAE). The best model is saved to S3 and can be used for predictions in other scripts.
   - **Note:** There is a separate branch named `sagemaker-training` for executing a training job using AWS SageMaker instead of training locally.

3. **`eval`**: Contains scripts to evaluate the model.
   - **`eval.py`**: Generates plots on a test set (data the model has never seen) to visualize the relationship between actual Ethereum prices and predicted prices.

## Skills Demonstrated

- **Data Engineering**: Automated data retrieval and preprocessing using Python scripts. This includes data extraction from external sources (Polygon API), transforming raw data into structured formats, and storing it in AWS S3.

- **MLOps and Automation**: Utilized GitHub Actions to schedule and automate data collection tasks on a weekly basis. Demonstrated the ability to integrate continuous integration and deployment (CI/CD) practices into machine learning workflows.

- **Deep Learning**: Developed and tuned an LSTM model for time series forecasting. Employed Keras Tuner to perform hyperparameter optimization to minimize validation loss (MAE).

- **Model Training and Deployment**: Implemented training scripts for both local and cloud-based (AWS SageMaker) environments, showcasing flexibility in training model deployment strategies and cloud computing.

- **AWS Cloud Services**: Leveraged AWS services such as S3 for data storage and SageMaker for scalable model training. Managed sensitive data and credentials securely with AWS Secrets Manager.

- **Statistical Analysis and Visualization**: Created visualizations to evaluate model performance, including error distributions and confidence intervals, to effectively communicate insights derived from model predictions.

- **Version Control and Collaboration**: Managed a multi-branch repository, utilizing branching strategies (e.g., `sagemaker-training` branch) to support different training environments and collaboration on the project.

- **Time Series Forecasting**: Demonstrated expertise in time series analysis by processing and forecasting Ethereum price data using deep learning models, handling challenges specific to financial data like non-stationarity and volatility.
