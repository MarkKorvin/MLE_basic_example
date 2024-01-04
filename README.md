# MLE_basic_example
This is the template for the well-structured ML project


docker build -f ./training/Dockerfile -t training_image . 

docker build -f ./inference/Dockerfile --build-arg model_name=prod_model.pickle -t training_image  .