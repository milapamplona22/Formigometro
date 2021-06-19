# Docker

Atualmente, o opencv para uso com gpu (módulo DNN) precisa ser compilado manualmente. Para tal, é bom usar o Docker. Optamos por usar o módulo dnn para rodar inferência da yolo para poder eliminar a dependência do darknet dentro desse docker. No entanto, a yolov5 passou a utilizar o pytorch ao invés de darknet. Eu tentei exportar uma yolov5 para ONNX para usar no opencv::dnn, mas dá erro. A yolov5 até é exportada, mas na hora de rodar a inferência em opencv deu erro. Por outro lado, a yolov5 já distribui um dockerfile que funciona bem, foi mais fácil alterar o código python da inferência.

Temos o seguinte cenário: quando rodamos a yolov3, dependemos do opencv::dnn, e portanto de uma compilação completa local, na qual vale a pena usar docker. No caso de uso da yolov5, fazemos a inferência direto em pytorch. Portanto, dependemos um pouco do futuro da yolo. No momento, com a saída do Joseph Redmond, o darknet não tem futuro. Usar diretamente o torch parece ser o caminho para não depender do opencv customizado. Ultimamente é mais fácil instalar o torch que compilar o opencv para gpu.

Além disso tudo, é curiosamente muito, muito trabalhoso manter um docker que alinhe (nvidia driver) -> cuda e cudnn -> tensorflow -> opencv. É mais fácil usar um dos container (docker pull) do datamachines. https://github.com/datamachines/cuda_tensorflow_opencv

Como compilar opencv do zero com suporte a gpu e módulo dnn
https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/

