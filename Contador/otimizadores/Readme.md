ultima atualização 19.06.2021
# Otimizadores
Otimização de processos. Atualmente usa o bayesian optimization. Paraleliza a execução entre as amostras sob um mesmo conjunto de parâmetros.


- input = tabela .csv :
    +   1a coluna: vídeo, ou input do comando a ser executado
    +   2a coluna: respectivo dado de rótulo (ground truth)

No caso atual, pareamos a saída do detector (../yolo/yoloV5Detect.py) e anotação manual do cvat (.xml) de cada vídeo, e como parâmetros a serem otimizados os parâmetros do rransac. 

A especialização da task, na verdade, deveria ser realizada fora da pasta do otimizador, como uma especialização para cada caso. No entanto, como modelo de exemplo será mantido o caso atual como exemplo (task.py)

*otmMngr.py*
o otimizador a ser utilizado. 

*runTasks.py*
executa um programa como chamado pela linha de comando, com a possibilidade de se estabelecer um timeout e de se submeter a paralizações


*MOTmetrics.py*
usa o pacote MOTmetrics do python para cálculo de erros de tracking
