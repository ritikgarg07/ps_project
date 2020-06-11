## ps_project
# PS Project for summer 2020 at IIRS

### Setting up development environment 

`git clone https://github.com/ritikgarg07/ps_project.git`  
`cd ps_project`  
`sh scripts/setup.sh`   (This will create a docker image with the required packages)      
`sh scripts/download.sh -d -u`  
  
### NOTE:   
    For download.sh 2 flags are available:   
        if -d is provided it will download dataset  
        if -u is provided it will unzip dataset  
    You can manually unzip dataset and place it in ps_project/data/raw/ and skip these flags
    
        



