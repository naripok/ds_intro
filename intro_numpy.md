# Numpy

O [Numpy](http://www.numpy.org/) é a biblioteca matemática padrão do python. Suas rotinas são compiladas em C, Fortran e outras linguagens mais rapidas do que Python, fazendo com que a computação de grande volume de dados seja feita de forma rápida e eficiênte. Sua estrutura de dados básica é o array.


```python
import numpy as np # importa o modulo para uso

array_1 = np.array([i for i in range(1,5)]) # cria o array a partir de uma lista
array_1
```




    array([1, 2, 3, 4])



Um array pode ser modificado em elementos, mas não em tamanho como uma lista. Selecionar elementos por indice também é possivel.


```python
array_1[0] # elemento 0 do array
```




    1




```python
array_1[1] = 5 # modifica o elemento 1 do array para '5'
array_1
```




    array([1, 5, 3, 4])



O numpy realiza operação com seus metodos proprios, como por exemplo:


```python
array_2 = np.multiply(2, array_1) # multiplica um array por um escalar
array_2
```




    array([ 2, 10,  6,  8])




```python
array_3 = np.divide(array_1, array_2) # divide um array pelo outro
array_3
```




    array([0.5, 0.5, 0.5, 0.5])




```python
array_4 = np.dot(array_1, array_2.T) # transpoe o array 2 e realiza multiplicaçao matricial (dot)
array_4
```




    102




```python
array_5 = array_3.reshape([-1,2]) # redimensiona o array em uma matriz 2x2
array_5
```




    array([[0.5, 0.5],
           [0.5, 0.5]])



O Numpy é um modulo muito poderoso e não pretendo demonstrar todas suas funcionalidades aqui. Para maiores informações, recorrer a [documentação](https://docs.scipy.org/doc/numpy/) ou ao seu melhor amigo [Stack Overflow](https://stackoverflow.com/).

