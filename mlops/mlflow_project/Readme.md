# Getting started with MLflow

Start by creating an environment

```bash
conda create -n exp-tracking-env python=3.9
```

Activate the environment

```bash
conda activate exp-tracking-env
```

Install the required packages

```bash
pip install -r requirements.txt
```

Start the mlflow server with the following command.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
This command will start the mlflow server on port 5000. You can access the mlflow server at http://localhost:5000 and we mentioned to mlflow that we want to use sqlite database file mlflow.db which will be used to store the experiment data.

![Alt text](image.png)