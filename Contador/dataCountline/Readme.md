19.06.2021

Acho que a countline é complicada demais para o caso. Analisar em as trajetórias, offline, 1 a 1 e checar em que momentos y > ycountline e y < ycountline seria mais prático e daria um output mais interpretável.
É bastante simples, poderia fazer do zero

---

10/2018
## Glossário:
- countline: conjunto de segmentos
- segmento: unidade (trecho) da countline
- pontos limite: (p1,p2) pontos que definem um segmento
- ponto externo: (q1,q2) pontos que serão dados para a countline ver se houve cruzamento ou não
- trajetória: sequência sucessiva de pontos externos

---

## Uso:
>```
Countline cline({cv::Point(x1,y1), cv::Point(x2,y2), ...})
Countline cline({x1,y1,x2,y2,...}
// os segmentos serão: p1->p2, p2->p3, ...
// contagem a cada par de pontos sucessivos
cline.update(vector<cv::Point>trajetoria)
// contagem desconsiderando u-turn
cline.update({trajetoria[0], trajetoria[-1]}) // (-1: último elemento)
```

O testeCountline.cpp permite que se desenhe com o mouse uma trajetória sobre uma countline e, apertando 'c', ele conta e imprime o resultado.

---

## To Do:
1. Inserir int Countline::n (quantidade de segmentos da countline)


## Problemas:
### 1. Quando um ponto externo cai sobre a reta
[Atualmente Conta]. Quando uma trajetória tem um ponto que cai justamente sobre um segmento, por exemplo:
> segmento s = [x1=0, y1=120, x2=320, y2=120]
> 
> trajetória t = [(100,100), (100,120), (100,130)]

Se a trajetória for analisada de dois em dois pontos sucessivos, a trajetória cruzará a reta duas vezes: tanto no trecho [(100,100),(100,120)] quanto no trecho seguinte [(100,120), (100,130)], pois o ponto (100,120) cai justamente sobre o segmento da countline.

Não contar quando o ponto externo cai sobre a reta não resolve porque, no mesmo caso, ele não produziria nenhuma contagem.

Passando-se para a countline somente o primeiro e o último ponto de uma trajetória - método que também resolve contagem de meias voltas - virtualmente resolve a questão (a não ser que o primeiro ou último ponto caia sobre o segmento, mas a probabilidade é menor dado que a countline geralmente é mais central).

### 2. Linha 90o 180o
Quando tem uma linha (segmento) reta de ângulos mútiplos de 90o - em que x1==x2 ou y1==y2 - todos os pontos do segmento (inclusive os inetrmediários entre os pontos limite) estão sobre números inteiros. Isso facilita que pontos externos possam cair sobre a countline. Iserir uma inclinação na reta (ex: y1=120, y2=121) faz com que os pontos intermediários do segmento sejam floats e, portanto, fica mais difícil de um ponto externo cair sobre a countline.

# Log:
- 10/2018 - criação
- 10/2018 - inserido void Countline::getTotals()