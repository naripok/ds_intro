# Introdução ao Python

Python é uma linguagem de programação muito popular na comunidade científica. Sendo um projeto *open-source* e apoiada por uma imensa comunidade, a linguagem esta em constante desenvolvimento. Uma ampla gama de modulos de terceiros esta disponível para python, sendo este um dos grandes fortes da linguagem. Você não precisa fazer tudo de novo do zero quando ja fizeram grande parte do trabalho para você e este trabalho foi realizado pelos melhores ciêntistas e programadores do mundo. A linguagem é legivel, com manejo automatico de memória, e é interpretada, o que quer dizer que você não precisará compilar seus programas antes de testar.

Para utilizar o python no jupyter notebook, digite o codigo na celula *Code* e entao execute-a (CTRL+ENTER).

Vamos a um exemplo básico:


```python
print("Hello World!")
```

    Hello World!


O código acima é um comando simples de "*print*", e tem como saída o argumento passado para esta função impresso no *stdout*.

Para definir variáveis, usa-se o sinal '=':


```python
var = 42
```

No jupyter, caso a ultima linha da celula for apenas o nome de um dado objeto, seu valor será mostrado abaixo logo após a execução da celula.


```python
var
```




    42



*Loops* basicos e estruturas de controle são intuitivos em python.

*Loop while*:


```python
i = 0
while i <= 3:
    print(i)
    i += 1
```

    0
    1
    2
    3


*Loop for*:


```python
for i in range(3):
    print(i)
```

    0
    1
    2


Vale ressaltar que a maioria dos intervalos definidos em python são abertos a direita. Pode-se notar esta propriedade no *loop for* acima. 

*If, elif, else:*


```python
for i in range(4):
    if i == 0:
        print("executou if, i = %d" % i)
    elif i == 1:
        print("executou elif, i = %d" % i)
    else:
        print("executou else, i = %d" % i)
```

    executou if, i = 0
    executou elif, i = 1
    executou else, i = 2
    executou else, i = 3


Estrutura *try*, *except* para captura de erros:


```python
ex = [1,2]
for i in range(3):
    try:
        print(ex[i])
    except Exception as e:
        print(e)
```

    1
    2
    list index out of range


As estruturas de dados mais comuns em python são listas, dicionários (os dois mutáveis) e tuples (imutáveis).


```python
obj_1 = 0
obj_2 = 1
obj_3 = 2

list_1 = [obj_1, obj_2, obj_3] # lista

dict_1 = {'key_1': obj_1, 'key_2': obj_2, 'key_3': obj_3} # dicionario

tuple_1 = (obj_1, obj_2, obj_3) # tuple
```

Listas são estruturas de ordem definida, ou seja, seus elementos conservam a ordem em que foram integrados, e estas tem o seguinte formato:


```python
list_1
```




    [0, 1, 2]



Elementos da lista podem ser selecionados individualmente ou em "fatias", como por exemplo:


```python
list_1[0], list_1[1] # index
```




    (0, 1)




```python
list_1[1:3] # Slice
```




    [1, 2]



Dicionários são estruturas sem ordem definida, e se apresentam como baixo:


```python
dict_1
```




    {'key_1': 0, 'key_2': 1, 'key_3': 2}



Elementos de dicionários são selecionados chamando por sua "chave":


```python
dict_1['key_1']
```




    0



Pode-se ainda realizar operações em cada par chave-elemento individualmente com o *loop for*:


```python
for key, item in dict_1.items():
    print(key, item)
```

    key_1 0
    key_2 1
    key_3 2


*Tuples* são elementos com ordem definida e se apresentam como abaixo:


```python
tuple_1
```




    (0, 1, 2)



Listas e dicionários são mutáveis, e pode-se adicionar ou retirar objetos dinamicamente com os comandos apropriados, já para modificar uma tuple, a mesma deve ser destruída e uma nova deve ser colocada no lugar.
A [documentação](https://docs.python.org/3.5/) do python contém o guia completo sobre estruturas de dados e outras particularidades.


```python
list_1.append(3)
list_1
```




    [0, 1, 2, 3]




```python
dict_1['key_3'] = 3
dict_1
```




    {'key_1': 0, 'key_2': 1, 'key_3': 3}




```python
tuple_1
```




    (0, 1, 2)




```python
tuple_1 = (0,1,2,3)
tuple_1
```




    (0, 1, 2, 3)



Além disso, temos *sets*, imutáveis e de elementos únicos:


```python
set([1,2,3,4,4,4,4])
```




    {1, 2, 3, 4}



Operações matemáticas simples podem ser executadas diretamente em python, como abaixo:


```python
1 + 2
```




    3




```python
2 * 3
```




    6




```python
2 / 3
```




    0.6666666666666666




```python
2**2 # Exponenciação, 2²
```




    4



Para realizar processos mais complexos, recomenda-se a utilização do *Numpy*.

Para um tutorial completo mais didático do que a documentação, recomendo o tutorial [A byte of python](https://python.swaroopch.com/basics.html).
