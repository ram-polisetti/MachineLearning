# MLOPS ZoomCamp

## Experiment Tracking

### Important Concepts

- *ML Experiment:* A sequence of steps that you take to train a model.(The process of building an ML model)
- *Experiment Run:* Each trial in an ML experiment is called a run.
- *Run Artifacts:* Data, Models, Metrics, Parameters, Source Code, etc.(Any file that is assciated with am ML run)
- *Experiment Metadata:* Data about the experiment itself. (When was the experiment run? Who ran it? What code was used? What were the parameters? What were the results?)

### What is Experiment Tracking?

- Experiment tracking is the process of tracking all the relevant information from an ML experiment.
  - Source code
  - Environment
  - Data
  - Model
  - Hyperparameters
  - Metrics
  - ...
  
### Why is Experiment Tracking Important?

- *Reproducibility* and *Replicability*
- *Organization* and *Collaboration*
- *Optimaztion* and *Debugging*

### Experiment Tracking Tool - [MLFlow](https://mlflow.org/)

- MLflow is a tool to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.
- It has four main modules:
  - MLflow Tracking -> Experiment Tracking
  - MLflow Models -> A model packaging format and tools that let you easily deploy the same model (from any ML library) to batch and real-time scoring on platforms such as Docker, Apache Spark, Azure ML and AWS SageMaker.
  - MLflow Registry -> Model registry for collaborative model management
  - MLflow Projects -> Packaging format for reproducible runs on any platform
  
#### MLflow Tracking

This allows you to organize your experiments into runs, and keep track of:

- parameters
- metrics
- artifacts -> any file like visualizations, the input file or the output file etc., that is associated with the run which you want to take a look at later.
- Metadata

Along with this information, MLflow Tracking also records:

- source code
- Verison of the code
- Start and End time
- Author

