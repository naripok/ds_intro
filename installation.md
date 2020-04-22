# Introdução

Esta é a sessão introdutória onde realizaremos o *setup* do ambiente de desenvolvimento e seus módulos e dependências.

Todos os passos descritos a seguir assumem que o sistema operacional utilizado é o **Ubuntu 18.04 LTS**.
Os passos são análogos para outros sistemas, e uma ampla gama de ótimos tutoriais de instalação esta disponivel na internet. Alternativamente, existe um pacote chamado [Anaconda](https://www.continuum.io/downloads) que oferece a maioria das ferramentas para ciência de dados em python e é de instalação facilitada. 

Durante este manual, as linhas prefixadas com `$` devem ser interpretadas como comandos a serem inseridos e executados no terminal. Por exemplo:
```
$ python3
```

Caso algo dê errado, `Ctrl + C` cancela a execução do comando e restaura o controle do terminal.


## Ambientes virtuais

Neste manual serão utilizados como linguagem de programação o Python 3.4+, e como IDE o Jupyter Notebook.

Estão disponíveis na internet ótimos ambientes virtuais de desenvolvimento
hosteados em nuvem, de forma que não é necessária a instalação de um 
ambiente local de desenvolvimento.
Exemplos desses ambientes virtuais são o
 [Kaggle](https://www.kaggle.com/notebooks/welcome) 
e o 
[Google Colab](https://colab.research.google.com/).

Se você escolher por produzir um ambiente de desenvolvimento local, os 
seguintes passos devem ser tomados:

### Instalação

Os seguintes módulos serão utilizados para a manipulação numérica e visualizações:

- [Numpy](http://www.numpy.org/), como biblioteca matemática.
- [Pandas](http://pandas.pydata.org/), como ferramenta para análise de dados.
- [Sklearn](http://scikit-learn.org/stable/), como ferramenta de alto nível para modelagem e aprendizado de máquina.
- [Keras](https://keras.io/), como ferramenta de alto nível especifica para o desenvolvimento de redes neurais artificiais.
- [Matplotlib](https://matplotlib.org/), como ferramenta de visualização e graficos.
- [Seaborn](https://seaborn.pydata.org/), como ferramenta de visualização e graficos.
- [Statsmodels](https://www.statsmodels.org/stable/index.html), como ferramenta
de modelagem estatística

Os modulos serão instalados em um ambiente virtual de python separado da instalação do sistema, por sanidade.


### Checando a existência do python no sistema

Normalmente as distribuições linux vem com python instalado por padrão.
Para verificar, abra o terminal e digite o seguinte comando:
```
$ python3
```

Você deve ser recebido com um [REPL](https://colab.research.google.com/) do tipo:
```
Python 3.8.2 (default, Apr 8 2020, 14:31:25
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Para sair do shell, entre o seguinte comando e tecle `Enter`:
```
...
Type "help", "copyright",...
>>> exit()
```

Se você recebeu um shell como este, sua instalação do python já está pronta e
você pode pular os próximos passos, caso contrário, prosseguiremos para a instalação.


### Instalando o Python 3.x

Abra o terminal e digite os seguintes comandos:

```
$ sudo apt install software-properties-common
$ sudo add-apt-repository ppa:deadsnakes/ppa  
$ sudo apt update  
$ sudo apt install python3.7
$ sudo pip3 install --upgrade pip  
```

Feito isso, instale o virtualenv:

```
$ pip install virtualenv
```

Crie o ambiente virtual no diretorio desejado:

```
$ sudo virtualenv -p python3 .env
```

Ative o ambiente virtual:

```
$ source .env/bin/activate
```

Pronto. Digite no terminal o seguinte comando para verificar se a instalação foi um sucesso:

```
$ python --version
```

Este comando deve retornar a versão do python instalada no ambiente virtual, e deve ser igual a 3.X

Para sair do ambiente virtual, digite no terminal:

```
$ deactivate
```


## Instalalando o Jupyter

Com o pip, o gerenciador de pacotes do python, ja instalado, é facil prosseguir com as instalações.
Para instalar o jupyter, digite no terminal enquanto com o ambiente virtual ativado:

```
(.env) $ pip install jupyter
```

Quando finalizada a instalação, o jupyter pode ser acessado de dentro do ambiente virtual com o comando:

```
(.env) $ jupyter notebook
```

Isto abrirá uma janela de brower com o jupyter explorer aberto. No canto superior direito clique em *new* -> python 3. Este comando abrirá um notebook vazio, este é seu ambiente de trabalho. A [documentação](http://jupyter-notebook.readthedocs.io/en/latest/) deste projeto também é muito boa e esclarece maiores duvidas.

## Instalando os modulos científicos

Com o ambiente virtual ativado, digite no terminal:

```
(.env) $ pip install matplotlib seaborn numpy pandas sklearn tensorflow keras statsmodels pandas_datareader
```

Feito isso, você pode testar suas dependencias abrindo digitando no terminal:

```
(.env) $ python3
```

Isto abrira uma sessão interativa do python. Digite então:

```
>>> import matplotlib
>>> import seaborn
>>> import numpy
>>> import pandas
>>> import tensorflow
>>> import keras
>>> import sklearn
>>> import statsmodels
>>> import pandas_datareader
>>> exit()  
```

Caso todos os imports sejam executados sem retornar nenhum erro, sua instalação foi um sucesso.  
Ocorrendo algum erro, a documentação de cada um dos componentes contém manual completo de intalação em inglês.
