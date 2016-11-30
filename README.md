# LabCog #
## Repositório com códigos e exemplos para execução de análises estatísticas ##

1. Regras
2. Requerimentos
3. Uso básico


### 1. Regras ###
1. Manter o branch **master sempre funcional **
2. Quando for **alterar** algo existente ou **adicionar** algo novo, faça isso num **branch próprio**. Ao certificar-se de que este novo branch está funcionando e em versão final, só então dê merge com o branch master para disponibilizar á todos, e delete o branch que foi criado para o desenvolvimento
3. arquivos com **exemplos** de análises devem ser nomeados com 'exemplo_' no início
4. colocar **referências** sobre sites, livros, papers, tanto sobre a explicação da análise quanto sobre o código
5. se criar um exemplo em jupyter **notebook**, adicionar também o .html exportado



### 2. O quê é necessário? (requerimentos) ###

* instalar [git](https://git-scm.com)
* dependendo do código que for usar: python, R, matlab
* recomendável ter [Jupyter Notebook](http://jupyter.org) para executar os exemplos interativos (.ipynb)

### 3. Uso básico (= fazer uma cópia local)###
referência simples para usar o git http://rogerdudler.github.io/git-guide/

1. baixe a versão atual do repositório:
```
#!git
     git pull origin master
```
 Todas as alterações que você fizer serão somente realizadas na sua máquina.


2. Se você quiser jogar suas alterações fora e baixar de novo:
```
#!git
     git fetch --all
     git reset --hard origin/master
```
     