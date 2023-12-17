# Salary prediction project

This README provides instructions on how to set up a virtual environment for your machine learning project. It also includes instructions on how to build and run a Docker image for your project.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 (recommended)
- Pipenv (`pip install pipenv`)
- Docker (for building and running the Docker image)

## Setting Up the Virtual Environment

1. Clone this repository to your local machine:

   ```shell
   git clone git@github.com:kshilovskiy/ml_zoomcamp.git
   cd ml_zoomcamp/project_1
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
python app.py
```
The server will be available at `http://localhost:8000/`.

You can try the following request to test the prediction with curl:

```shell
curl --location --request POST 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{ "age": "24", "gender": "Male", "city": "Munich", "position": "Software Engineer", "experience_years": "5", "experience_years_germany": "2", "seniority_level": "Middle","vacation_days": "30","employment_status": "Full-time employee", "сontract_duration": "Unlimited contract", "language_at_work": "English", "company_size": "1000+", "company_type": "Product"}'
```



## Building and Running the Docker Image

1. Build the Docker image from the project directory (where the Dockerfile is located):

   ```shell
   docker build -t salary-predictor .
   ```

   Replace `your-image-name` with a suitable name for your Docker image.

2. Run the Docker container from the image:

   ```shell
   docker run -p 8080:8080 salary-predictor
   ```

   This command runs the container and maps port 8080 on your local machine to port 80 in the container. Adjust the ports as needed.

3. Your machine learning project should now be accessible at `http://localhost:8080` in your web browser.
4. You can try the following request to test the prediction with curl:

```shell 
curl --location --request POST 'http://localhost:8080/predict' \
--header 'Content-Type: application/json' \
--data-raw '{ "age": "24", "gender": "Male", "city": "Munich", "position": "Software Engineer", "experience_years": "5", "experience_years_germany": "2", "seniority_level": "Middle","vacation_days": "30","employment_status": "Full-time employee", "сontract_duration": "Unlimited contract", "language_at_work": "English", "company_size": "1000+", "company_type": "Product"}'
````
