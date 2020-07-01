# PS Project for summer 2020 at IIRS
# Hyperspectral Reconstruction from RGB Images

### Setting up environment 

`git clone https://github.com/ritikgarg07/ps_project.git`  
`cd ps_project`  
`sh scripts/setup.sh`              (This will create a docker image with the required packages)      
`sh scripts/download.sh -d -u`     (This will download, unzip and process the dataset)  
`cd ..`  
`sh ps_project/scripts/run.sh`     (This will create a docker container and mount the files appropriately)  
`cd /workspaces/ps_project`        (Make sure you're inside the container)  

### Training the model(s)

`python src/data_prepare.py`       (This will prepare the dataset for training)  
`python src/main.py`               (This will train and execute models)  
  
### Using the model for predictions

Store all the .bmp input rgb images in ps_project/sample/ and then run  
`python src/predict.py`

#### NOTE:   
For download.sh 2 flags are available:     
   if -d is provided it will download dataset      
   if -u is provided it will unzip dataset      
   You can manually download and unzip dataset and place it in ps_project/data/raw/ and run download.sh without any flags    
   Otherwise, you can manually download the dataset and place the zip in ps_project/ and use only the -u flag

### NOTE:    
   Specify the url link to the dataset in download.sh        

### NOTE:
   For changing which models to train and store, modify main.py

### NOTE: 
   For changing which model is used to predict data, modify predict.py
