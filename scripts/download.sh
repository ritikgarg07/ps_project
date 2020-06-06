#!bin/bash

wget -O ./data/data.zip https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip

unzip data.zip

# uncomment if you want to delete zip after extraction for space purposes
# rm data.zip