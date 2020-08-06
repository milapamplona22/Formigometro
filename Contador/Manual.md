## 1. Setup
usando docker
> `make cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0`


## 2. Rodar contagem

#### 2.1 Ver qual o id gerado pelo docker
`docker images`

ex da minha saída:
```
REPOSITORY                             TAG                             IMAGE ID            CREATED             SIZE
datamachines/cudnn_tensorflow_opencv   10.2_2.2.0_4.3.0-20200615       d58af4a715a4        14 hours ago        5.86GB
nvidia/cuda                            10.2-cudnn7-devel-ubuntu18.04   2b8bb5f68029        6 days ago          3.82GB
```

#### 2.2 Entrar na máquina gerada pelo docker (container)
entrar no docker (d58af4a715a4 é o ID da minha imagem, na minha máquina, em outra máquina pode ser outro valor, substituir de acordo no comando abaixo):

> `docker run -ti --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --ipc host --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm  -v /home/:/home/ d58af4a715a4 /bin/bash`

#### 2.3 Executar a análise de um vídeo
> `python yoloVideoDetect.py --video 4_2018-10-04_14-40-00.mp4 --config yolo-voc_1batch_v3.cfg --weights yolo-voc_1batch_final.weights  --classes mila.names --yml pedestrians.yml --show`

#### 2.4 Executar a análise de mais vídeos (em paralelo), com rush
> `find . -name "*.mp4" | ./rush -j 3 'python yoloVideoDetect.py --video {} --config yolo-voc_1batch_v3.cfg --weights yolo-voc_1batch_final.weights  --classes mila.names --yml {.}.yml'`

Nesse caso, como estamos na gpu, talvez não caibam muitas redes na mesma gpu ao mesmo tempo. Testando numa NVIDIA GTX 1050 Ti de 4GB de memória, consegui subir 3 ao mesmo tempo (ainda não testei com mais do que isso). Em gpus com menos memória, (a do lab tem 3GB), talvez tenha que testar para ver se é possível mesmo usar duas ou mais.

#### 2.4.b Vídeos em sequência (sem paralelismo), com e sem rush
##### 2.4.b.1 com rush
Se só quiser especificar para analisar vários vídeos em sequência, mas um por vez (sem paralelizar), é só substituir no comando acima o número de jobs por um (`-j 1`). 
##### 2.4.b.1 sem rush
Ou executar o comando acima sem o rush. Isso seria possível usando `find exec`, mas daí o comando muda um pouco

