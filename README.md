![oreilly-logo](images/oreilly.png)

# Hands on MLOps with PyTorch

This repository contains code for the [O'Reilly Live Online Training for Hands on MLOps with PyTorch](https://www.oreilly.com/live-events/hands-on-mlops-with-pytorch/0636920072575/0636920072574/)

This training is a hands-on look at the end-to-end Natural Language Processing pipeline with a case study focusing on model training and evaluation, deployment and model-serving in production, and combatting model drift. The session is mostly hands-on which means that a majority of it will be spent looking at code examples and running code to train and deploy state of the art NLP models.

We will use tools including TorchServe, TorchDrift, and mlflow to manage model versions and deploy them to Databricks, an industry leading Data Science and ML platform. We will also see several code examples throughout the training around an intent classification use-case using BERT to help solidify the theoretical concepts being introduced.

### Notebooks

[Model Training/Serving with BERT](notebooks/model_training.ipynb)

[Deteching Model Drift with TorchServe and Online Learning](notebooks/detecting_model_drift.ipynb)

[Deploying models with FastAPI](fastapi/)

[Deploying models with TorchServe](torchserve/)

[Cleaning Data](notebooks/data_cleaning.ipynb)

## Instructor

*Sinan Ozdemir* is currently the Director of Data Science at Directly, managing the AI and machine learning models that power the company’s intelligent customer support platform. Sinan is a former lecturer of Data Science at Johns Hopkins University and the author of multiple textbooks on data science and machine learning. Additionally, he is the founder of the recently acquired Kylie.ai, an enterprise-grade conversational AI platform with RPA capabilities. He holds a Master’s Degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, CA.