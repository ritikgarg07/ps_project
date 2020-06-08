#!bin/bash

while getopts ":du" opt; do
    case $opt in
    d)
        echo "-d was triggered! Downloading dataset"
        wget -O ./data/data.zip https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip      
        ;;
    u) 
        echo "-u was triggered! Unzipping dataset"
        unzip data.zip -d ./data/
        ;;
    \?)
        echo "Invalid option! No flags required if data already placed correctly"
        ;;
    esac
done


cd ./data

i=0
for f in */; do
    cd $f
    mv -v ./$f/* .
    rm -rf $f
    for f2 in *; do
        ext="${f2##*.}"
        if [ "$ext" = "db" ]; then
            mv $f2 $i.$ext
        elif [ "$ext" = "bmp" ]; then
            mv $f2 $i.$ext
        else
            name=`expr "/$f2" : '.*\(.\{2\}\)\.'`
            mv $f2 "${i}_${name}.${ext}"
        fi
    done
    cd ..
    mv $f $i
    echo $i
    i=$((i+1))
done
touch .gitkeep
# uncomment if you want to delete zip after extraction for space purposes
# cd .. 
# rm data.zip