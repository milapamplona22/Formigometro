# dataRRANSAC
última atualização: 19.06.2021

Executa o multi-target tracker RRANSAC (C++). O input é um yaml (ou yml) com as detecções, e a saída é outro yml que deve ser lido usando o dataRRANSACutils.py

## TL;DR
- como rodar: 
```
./dataRRANSAC -data exemploInput.yml -M 50 -U 5 -tau_rho 0.1 -tau_T 3 -Nw 100 -ell 200 -tauR 2 -Qm 2 -Rmx 5 -Rmy 17  -display ~/IncorGDrive/LabCog/ProjetoMila-andamento/dados/videosComplicados/1_2019-04-05_06-40-00.mp4 -out teste.yml
```
- build:
```
mkdir build
cd build
cmake ..
make
```
- esses parametros foram ajustados manualmente. As detecções de entrada foram obtidas com uma yolov5l. (Os parâmetros podem variar em outros casos. Neste em particular, o detector estava bastante bom) 
- os arquivos python (*.py) são para a leitura dos yml gerados pelo dataRRANSAC (C++)


## dataRRANSAC
O dataRRANSAC executa o rransac sobre um arquivo yml (Opencv) com os centroides. Os parâmetros do RRANSAC devem ser especificados na linha de comando. Os resultados produzem um outro yml (opencv)

* usa o rransacFrameN.hpp,cpp

##### Opções:
- data: arquivo yml de enrada
- out: yml de saída com as tracks geradas
- outfw: yml de saída com as tracks geradas, com as informações separadas em vetores pareados (frames, ids, position)
- display: (opcional), vídeo sobre o qual é mostrado o resultado do rransac
- parâmetros *obrigatórios* do RRANSAC:
```
 RRANSAC params: (must have)
  -M         (int) Number of stored models
  -U,        (float) Merge threshold
  -tau_rho   (float) Good model quality threshold
  -tau_T,    (int) Minimum number of time steps needed for a good model
  -Nw,       (int) Measurement window size
  -ell,      (int) Number of RANSAC iterations
  -tauR,     (int) Inlier threshold (multiplier)
  -Qm,       (float) Q multiplier
  -Rmx,      (float) Rx multiplier
  -Rmy       (float) Ry multiplier
```
*o output mais confiável atualmente é usar o outfw e depois lê-lo com o dataRRANSACutis.py com framewiseMeans2(readTrackerResults(yml))*


## dataRRANSACutils.py
função em python que lê o yml gerado pelo dataRRANSAC e devolve as tracks
* usa o pacote yaml. A melhor forma de usar é importar no seu código.
Além disso, ele tem uma função para criar um commando (cli) e rodar o dataRRANSAC a partir de um dict com os parâmetros 

## dataRRANSACyml2pkl.py
ToDo: converte as tracks de yml para um arquivo pkl.
* só mantive porque tem um exmplo de como ler o yml usando o opencv

## rransacFrameN
No rransac.hpp,cpp original (../code/), sempre que uma goodtrack se encerra ela fica disponível em std::<RRANSACmodel>tracks. Mas, apesar de saber a trajetória dela, não se pode saber outras informações como o N dos frames em que ela apareceu, o tamanho do blob e etc. Para resolver isso eu fiz um rransacFeats (formigometro), que insere dentro dos tracks métodos de obtenção de features desejadas e guarda os próprias valores dessas features. Mas uma classe features só pode ser especificada antes da própria classe do rransac para que possa ser inserida nele (fiz testes de como inserir uma classe base e depois derivar ela, mas o problema é fazer update dos valores, as iterações acabavam iniciando e acumulando instâncias na memória).

No caso em particular, a única coisa de fato necessária era saber o número do frame correspondente à cada posição de um track. Para isso fiz este rransacFeats específico que faz isso (rransacFrameN).

## ToDo:
- já fazer as médias de posição quando um id tem mais de uma posição no mesmo frame
- uniformizar isso para o out e para o outfw