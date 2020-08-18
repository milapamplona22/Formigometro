# dataRRANSAC

## dataRRANSAC
O dataRRANSAC executa o rransac sobre um arquivo yml (Opencv) com os centroides. Os parâmetros do RRANSAC devem ser especificados na linha de comando. Os resultados produzem um outro yml (opencv)


* usa o rransacFrameN.hpp,cpp

#### Opções:
-data: arquivo yml de enrada
-out: yml de saída com as tracks geradas
-display: (opcional), vídeo sobre o qual é mostrado o resultado do rransac

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


## dataRRANSACutils.py
função em python que lê o yml gerado pelo dataRRANSAC e devolve as tracks
* usa o pacote yaml

## dataRRANSACyml2pkl.py
ToDo: converte as tracks de yml para um arquivo pkl. 
* só mantive porque tem um exmplo de como ler o yml usando o opencv


## rransacFrameN
O rransac.hpp,cpp original (../code/) não mantém a identidade das tracks. Sempre que uma goodtrack se encerra, ela fica disponível em std::<RRANSACmodel>tracks, mas, apesar de saber a trajetória dela, não se pode saber outras informações como o N dos frames em que ela apareceu, o tamanho do blob e etc. Para resolver isso eu fiz um rransacFeats, que insere dentro dos tracks métodos de obtenção de features desejadas e guarda os próprias valores dessas features. Mas uma classe features só pode ser especificada antes da própria classe do rransac para que possa ser inserida nele (fiz testes de como inserir uma classe base e depois derivar ela,mas o problema é fazer update dos valores, as iterações acabavam iniciando e acumulando instâncias na memória).

No caso em particular a única coisa de fato necessária era saber o número do frame correspondente à cada posição de um track. Para isso fiz um rransac local que faz isso (rransacFrameN).
