# Automated Training Pipeline 

This repository provides an automated training pipeline that simplifies the process of training and updating deep learning models with new data.

The pipeline takes a common dataset from `torchvision.Dataset` (such as MNIST or FashionMNIST) and a configuration file in YAML format. 
It then automatically trains a simple CNN model on the specified data, compares its performance to the previous best model,
and saves the new model if it performs better.


## Motivation

One of the major things in AI/DL is the data. Often, we have a model and once we have "better" data, we retrain our 
models with the new data. However, this process may be cumbersome as 

- Have a VM instance (or local machine) and set up the environment (I admit, I do this manually too!) 
- We need to download the new data
- Train the model with the "better" data
- Get metrics, visualizations etc
- Save/Load/Verify if the model is functional/working fine
- Dockerize
- Deploy on cloud or where ever we want (I won't do this step)

So this repo will do the whole thing automatically. 


## Spoiler

Since I am not gonna collect more data, I will first get part of the data (say 50%) and do the training, save, dockerize etc. 
Then, I will gradually increase the amount of the data. Once certain conditions are met (e.g., the data size increase by x% 
or something like that), the code will do retraining. 

Users can change these conditions, for example, get the hash of the previous data. If the hash has changed, then retrain. 
One of the use case might be that the number of images may not change but the quality of it may change. So, user might wanna
run every midnight or once a week etc. This step is really not the most crucial aspect of the project


# Poor man's Jira

1. &check; Setting up the repo
2. &check; Data downloader for a bunch of classification tasks
3. &check; Simple CNN Model for classification
4. &check; Dataloaders (train, val, test)
5. &check; Full Training Pipeline (for a specific task)
6. Metric
7. Visualizations 
8. Save/load model 
9. Integrating YAML 
10. Automated training (From 50% of the data to 100% of the data)
11. Visualization export using plotly (Accuracy, Confusion metrics etc)
12. Making the training fully functional by only selecting project (MNIST/FashionMNIST etc)
13. Ray[Tune] integration 
14. Device-agnostic code (not needed for now as I have only cpu)
