Speech emotion recognition for the emo-db dataset (http://emodb.bilderbar.info/start.html).

1. To build the docker image

    docker build -t michele/ser .     

    This will setup the container with the required packages, download the data, augment it and run the flask and notebook server.

    If the data preparation is too long you can change the n_augm paramenters in src/data_preparation/config to 1. This will disable data augmentation. (Normally it's more convenient to augment samples on the fly but in the reference paper the authors don't follow this approach)

2. To run the container

    docker run -d -p 5000:5000 -p 8888:8888 --name ser michele/ser 

    Notice that we run in detached mode so that we can query the jupuyter notebook token. It might take a few seconds to start the jupyter notebook

    docker exec -it ser jupyter notebook list

3. Once logged in at localhost:8888 you can access the report and follow the analysis

4. For training the model you can request http://localhost:5000/train. Successive visits of the same url will return information about the progress of the training.

5. For predicting the emotion you request http://localhost:5000/predict?sample=<some_sample>
    You can get the list of samples in the database with http://localhost:5000/predict
    For example http://localhost:5000/predict?sample=16a07Fb.wav
