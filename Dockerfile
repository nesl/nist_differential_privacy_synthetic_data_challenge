FROM tensorflow/tensorflow:1.13.1-gpu-py3  
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "bash", "run_uclanesl.sh"]