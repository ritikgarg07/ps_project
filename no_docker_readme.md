# PS Project for summer 2020 at IIRS
# Hyperspectral Reconstruction from RGB Images

### Setting up environment  
(Assuming tensorflow gpu, and required nvidia libraries for utilising gpu are installed as per system)

`git clone https://github.com/ritikgarg07/ps_project.git`  
`cd ps_project`  
`sh scripts/download.sh -d -u` (This will download, unzip and process the dataset)  
`python3 -m pip install pillow`  
`python3 -m pip install matplotlib`  
`python3 -m pip install pyyaml`  

### Specify the base directory:
- Specify base_dir in config.yaml, date_prepare_h5.py, main.py, predict.py, sample_prepare.py and data.py

- There is only 1 location in each of the files and is in the top half itself. A comment has been added in all required files.

- base_dir is the ABSOLUTE path of the repository. 
For example: /home/username/ps_project/

### Training the model(s)

`python src/data_prepare_h5.py`       (This will prepare the dataset for training)  
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
