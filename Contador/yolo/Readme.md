# Yolo

Utilizamos a v3 e v5. 
Para a v3, usar o `yoloVideoDetect.py`, que requer o opencv compilado com gpu e módulo dnn.
Para a v5, usar o `yolov5Detect.py` com o ambiente do [yolov5](https://github.com/ultralytics/yolov5), seja no docker ou pip venv.


## Yolo V3
Linha de comando, o input é um vídeo, a saída é um yaml com as detecções.
A v3 tem algumas especificações que usamos como o cfg de 1 batch, usada para inferência. no caso, o yolo_v3_1batch_anchors.cfg disponível na pasta é uma yolov3 configurada para batch_size = 1 para a inferência de vídeo frame a frame. Além disso é necessário fornecer o arquivo de pesos e com o nome das categorias (.names)


## Yolo V5
Linha de comando, o input é um vídeo, a saída é um yaml com as detecções.
exemplo:
```
 python yolov5Detect.py --source ../../yolov5/1_2019-04-04_23-00-00.mp4 --weights ../../yolov5/weights/yolov5l_best_50.pt --conf-thres 0.1 --view-img --nosave --yml teste.yml --yolov5_path ../../yolov5
```
Eu inseri a saída em formato yml, o mesmo usado no restante do nosso pipeline. Precisa indicar onde você instalou o repositório do [yolov5](https://github.com/ultralytics/yolov5) no argumento `--yolov5_path`, ele depende disso. Caso queira exportar o vídeo com as detecções, remova o --nosave
como no caso testado o detector estava com raríssimos falsos positivos usei o --conf-thresh 0.1