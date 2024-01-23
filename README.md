This project illustrates a simple, yet effective template to organize an ML project. Following good practices and principles, it ensures a smooth transition from model development to deployment.


## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── iris_train.csv
│   └── iris_infer.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.

## Data:
Data is the cornerstone of any Machine Learning project. For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.
- The data files are included in the package - as it was required in the task description, so there is no need to run this script.
- The data folder also holds the training_scaler.gz file that is a packaged version of the scaler fit on the training data that will be used during inference. (This file itself is not data, but generated for the data present in the folder. The scaler is also generated during the building of the training image, so regenerating the data overwrites this file.)

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```

To run the training, get metrics on the trained model quality and copy the data files, packaged scaler and the trained model itself from the container to the local env:
```bash
docker run -v ./models:/app/models -v ./data:/app/data training_image

```

2. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```
If you choose to do so, you will need to:
- set up a virtual environment
- install packages from requirements.txt
- run the data_process/data_generation.py manually
before runnin the actual training/train.py script.


## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image with the model you like:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name.pt> --build-arg settings_name=settings.json -t inference_image .
```
If not sure about available built models, run:
`ls models/`

- Run the inference Docker container:
```bash
docker run -v ./models:/app/models -v ./data:/app/input -v ./results:/app/results inference_image
```
  
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```
If you did so for the training step, this should work without further preparation.  