FROM tensorflow/tensorflow:latest-gpu

 RUN apt-get update && \
     apt-get install git -y
     
RUN bash