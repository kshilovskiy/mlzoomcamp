# Smoker prediction project

This README provides instructions on how to set up a virtual environment for your machine learning project. It also includes instructions on how to build and run a Docker image for your project.

## Problem
The problem is taken from the kaggle competition [Binary Prediction of Smoker Status using Bio-Signals](https://www.kaggle.com/competitions/playground-series-s3e24).
The challenge states that a team of scientists is working on a machine learning project to predict whether someone smokes or not. 
The goal of this project is to help them create a model that can figure out a person's smoking status using various health data.

### Health Dataset Overview

- **Age:** 5-year groups
- **Height:** (in centimeters)
- **Weight:** (in kilograms)
- **Waist Circumference**
- **Eyesight:**
   - Left eye
   - Right eye
- **Hearing Ability:**
   - Left ear
   - Right ear
- **Blood Pressure:**
   - Systolic
   - Diastolic
- **Fasting Blood Sugar Levels**
- **Cholesterol Levels:**
   - Total
   - HDL (High-Density Lipoprotein)
   - LDL (Low-Density Lipoprotein)
   - Triglycerides
- **Hemoglobin Levels**
- **Urine Protein Concentration**
- **Serum Creatinine Levels**
- **AST (Glutamic Oxaloacetic Transaminase) and ALT (Glutamic Pyruvic Transaminase) Types**
- **γ-GTP (Gamma-Glutamyltransferase) Levels**
- **Presence of Dental Caries**
- **Smoking Status**



## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 (recommended)
- Pipenv (`pip install pipenv`)
- Docker (for building and running the Docker image)

## Setting Up the Virtual Environment

1. Clone this repository to your local machine:

   ```shell
   git clone git@github.com:kshilovskiy/mlzoomcamp.git
   cd mlzoomcamp/projects/capstone_1
   ```

2. Create a virtual environment with Pipenv:

   ```shell
   pipenv install
   ```

   This command will create a virtual environment and install the required Python packages listed in the `Pipfile`.

3. Activate the virtual environment:

   ```shell
   pipenv shell
   ```

   Your terminal prompt should now indicate that you are in the virtual environment.

## Deactivating the Virtual Environment

To deactivate the virtual environment and return to your system's global Python environment, use:

```shell
exit
```

## Building the model
To build the model, run the following command:
```shell
python train.py
```

Model and the vectorizer will be saved in the `model` folder.


## Running Flask Application

To run the Flask application, run the following command:
```shell
python predict.py
```
The server will be available at `http://localhost:8000/`.

You can try the following request to test the prediction with curl:

```shell
curl --location --request POST 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
               "age": 40.0,
               "height(cm)": 175.0,
               "weight(kg)": 65.0,
               "waist(cm)": 85.0,
               "eyesight(left)": 1.5,
               "eyesight(right)": 1.5,
               "hearing(left)": 1.0,
               "hearing(right)": 1.0,
               "systolic": 116.0,
               "relaxation": 74.0,
               "fasting blood sugar": 92.0,
               "Cholesterol": 179.0,
               "triglyceride": 206.0,
               "HDL": 41.0,
               "LDL": 97.0,
               "hemoglobin": 15.1,
               "Urine protein": 1.0,
               "serum creatinine": 1.1,
               "AST": 23.0,
               "ALT": 35.0,
               "Gtp": 56.0,
               "dental caries": 0.0}'
```

## Building and Running the Docker Image

1. Build the Docker image from the project directory (where the Dockerfile is located):

   ```shell
   docker build -t smoking-predictor .
   ```

2. Run the Docker container from the image:

   ```shell
   docker run -p 8080:8080 smoking-predictor
   ```

   This command runs the container and maps port 8080 on your local machine to port 8080 in the container. Adjust the ports as needed.

3. Your machine learning project should now be accessible at `http://localhost:8080` in your web browser.
4. You can try the following request to test the prediction with curl:

```shell 
curl --location --request POST 'http://localhost:8080/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
               "age": 40.0,
               "height(cm)": 175.0,
               "weight(kg)": 65.0,
               "waist(cm)": 85.0,
               "eyesight(left)": 1.5,
               "eyesight(right)": 1.5,
               "hearing(left)": 1.0,
               "hearing(right)": 1.0,
               "systolic": 116.0,
               "relaxation": 74.0,
               "fasting blood sugar": 92.0,
               "Cholesterol": 179.0,
               "triglyceride": 206.0,
               "HDL": 41.0,
               "LDL": 97.0,
               "hemoglobin": 15.1,
               "Urine protein": 1.0,
               "serum creatinine": 1.1,
               "AST": 23.0,
               "ALT": 35.0,
               "Gtp": 56.0,
               "dental caries": 0.0}'
````

## Deploying the Docker Image to AWS Elastic Beanstalk
1. Install Elastic Beanstalk CLI:
```shell
pipenv install awsebcli
# To have 'eb' command available
 pipenv shell
```
2. Initialize the directory with the Elastic Beanstalk CLI:
```shell 
eb init -p "Docker running on 64bit Amazon Linux 2" smoking-predictor-eb -r eu-central-1
```
3. Run the project locally:
```shell
eb local run --port 8080 
```
4. Deploy the project to AWS:
```shell
eb create smoking-predictor-eb-env
```

## Serving Predictions
The project is deployed to AWS Elastic Beanstalk and is available at http://konstantin-predictor-env.eba-32ypi5nq.eu-central-1.elasticbeanstalk.com/predict.

Here is how you can try it oue with curl:
```shell
curl --location --request POST 'http://konstantin-predictor-env.eba-32ypi5nq.eu-central-1.elasticbeanstalk.com/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
               "age": 40.0,
               "height(cm)": 175.0,
               "weight(kg)": 65.0,
               "waist(cm)": 85.0,
               "eyesight(left)": 1.5,
               "eyesight(right)": 1.5,
               "hearing(left)": 1.0,
               "hearing(right)": 1.0,
               "systolic": 116.0,
               "relaxation": 74.0,
               "fasting blood sugar": 92.0,
               "Cholesterol": 179.0,
               "triglyceride": 206.0,
               "HDL": 41.0,
               "LDL": 97.0,
               "hemoglobin": 15.1,
               "Urine protein": 1.0,
               "serum creatinine": 1.1,
               "AST": 23.0,
               "ALT": 35.0,
               "Gtp": 56.0,
               "dental caries": 0.0}'
```

The response returns the probability of a person being a smoker:
```json
{
    "prediction": "[0.8566376]"
}
```

