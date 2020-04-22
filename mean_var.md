# Medias, Moda e Mediana, Medidas de Tendência

Neste capitulo, discutiremos maneiras de resumir um conjunto de dados usando um único número. O objetivo é capturar informações sobre a distribuição de dados.

## Média aritmética

A média aritmética é usada com freqüência para resumir dados numéricos, e geralmente é o que se supõe significar pela palavra "média". É definida como a soma das observações divididas pelo número de observações:
$$ \mu = \frac {\sum_ {i = 1} ^ N X_i} {N} $$

onde $ X_1, X_2, \ldots, X_N $ são nossas observações.


```python
# Two useful statistical libraries
import scipy.stats as stats
import numpy as np

# We'll use these two data sets as examples
x1 = [1, 2, 2, 3, 4, 5, 5, 7]
x2 = x1 + [100]

print('Média de x1:', sum(x1), '/', len(x1), '=', np.mean(x1))
print('Média de x2:', sum(x2), '/', len(x2), '=', np.mean(x2))
```

    Média de x1: 29 / 8 = 3.625
    Média de x2: 129 / 9 = 14.333333333333334


Também podemos definir uma média aritmética *ponderada*, que é útil para especificar explicitamente o número de vezes que cada observação deve ser contada. Por exemplo, ao computar o valor médio de um portfólio, é mais conveniente dizer que 70% das suas ações são do tipo X ao invés de fazer uma lista de cada ação que você detém.

A média aritmética ponderada é definida como
$$ \sum_ {i = 1} ^ n w_i X_i $$

onde $ \sum_ {i = 1} ^ n w_i = 1 $. Na média aritmética usual, temos $ w_i = 1 /n $ para todo $ i $.

## Mediana

A mediana de um conjunto de dados é o número que aparece no meio da lista quando ele é ordenado em ordem crescente ou decrescente. Quando temos um número ímpar $ n $ de pontos de dados, este é simplesmente o valor na posição $ (n + 1) / 2 $. Quando temos um número par de pontos de dados, a lista se divide pela metade e não há nenhum item no meio; então definimos a mediana como a média dos valores nas posições $ n / 2 $ e $ (n + 2) / 2 $.

A mediana é menos afetada por valores extremos nos dados do que a média aritmética. Ele nos informa o valor que divide o conjunto de dados ao meio, mas não o quao menores ou maiores são os outros valores.


```python
print('Mediana de x1:', np.median(x1))
print('Mediana de x2:', np.median(x2))
```

    Mediana de x1: 3.5
    Mediana de x2: 4.0


## Moda

A moda é o valor mais freqüente em um conjunto de dados. Pode ser aplicado a dados não-numéricos, ao contrário da média e da mediana. Uma situação em que é útil é para dados cujos valores possíveis são independentes. Por exemplo, nos resultados de um dado (do tipo que se joga em um cassino) ponderado, tirar 6 muitas vezes não significa que é provável que o próximo resultado seja 5; então, saber que o conjunto de dados tem uma moda 6 é mais útil do que saber que tem uma média de 4,5.


```python
# Scipy has a built-in mode function, but it will return exactly one value
# even if two values occur the same number of times, or if no value appears more than once
print('Moda de x1:', stats.mode(x1)[0][0])

# So we will write our own
def mode(l):
    # Count the number of times each element appears in the list
    counts = {}
    for e in l:
        if e in counts:
            counts[e] += 1
        else:
            counts[e] = 1
            
    # Return the elements that appear the most times
    maxcount = 0
    modes = {}
    for (key, value) in counts.items():
        if value > maxcount:
            maxcount = value
            modes = {key}
        elif value == maxcount:
            modes.add(key)
            
    if maxcount > 1 or len(l) == 1:
        return list(modes)
    return('Moda não definida')
    
print('Todas as modas de x1:', mode(x1))
```

    Moda de x1: 2
    Todas as modas de x1: [2, 5]


Para os dados que podem assumir muitos valores diferentes, como em variáveis continuas, pode não haver valores que aparecem mais de uma vez. Neste caso, podemos classificar intervalos de valores, como fazemos quando construímos um histograma e, em seguida, encontrar a moda do conjunto de dados em que cada valor é substituído pelo valor da sua classe. Ou seja, descobrimos quais das classes ocorre na maioria das vezes.


```python
variable = np.random.random(100)
print('Moda da variável:', mode(variable))

# Since all of the samples are distinct, we use a frequency distribution to get an alternative mode.
# np.histogram returns the frequency distribution over the bins as well as the endpoints of the bins
hist, bins = np.histogram(variable, 20) # Break data up into 20 bins
maxfreq = max(hist)
# Find all of the bins that are hit with frequency maxfreq, then print the intervals corresponding to them
print('Moda das classes:', [(bins[i], bins[i+1]) for i, j in enumerate(hist) if j == maxfreq])
```

    Moda da variável: Moda não definida
    Moda das classes: [(0.8998283354848982, 0.9493160608322376)]


## Média Geométrica

Enquanto a média aritmética utiliza a adição, a média geométrica usa a multiplicação:
$$ G = \sqrt [n] {X_1X_1 \ldots X_n} $$

para observações $ X_i \geq 0 $. Também podemos reescrevê-la como uma média aritmética usando logaritmos:
$$ \ln G = \frac {\sum_ {i = 1} ^ n \ln X_i} {n} $$

A média geométrica é sempre menor ou igual à média aritmética (quando se trabalha com observações não-negativas), com igualdade somente quando todas as observações são iguais.



```python
# Use scipy's gmean function to compute the geometric mean
print('Média geométrica de x1:', stats.gmean(x1))
print('Média geométrica de x2:', stats.gmean(x2))
```

    Média geométrica de x1: 3.0941040249774403
    Média geométrica de x2: 4.552534587620071


## Média Harmônica

A média harmônica é menos comum que os outros tipos de médias. É definido como
$$ H = \frac {n} {\sum_ {i = 1} ^ n \frac {1} {X_i}} $$

Tal como acontece com a média geométrica, podemos reescrever a média harmônica para parecer uma média aritmética. O inverso da média harmônica é a média aritmética dos inverso das observações:
$$ \frac {1} {H} = \frac {\sum_ {i = 1} ^ n \frac {1} {X_i}} {n} $$


A média harmônica para números não negativos $ X_i $ é sempre no máximo a média geométrica (que é no máximo a média aritmética), e eles são iguais somente quando todas as observações são iguais.


```python
print('Média harmônica de x1:', stats.hmean(x1))
print('Média harmônica de x2:', stats.hmean(x2))
```

    Média harmônica de x1: 2.5590251332825593
    Média harmônica de x2: 2.869723656240511


A média harmônica pode ser usada quando os dados podem ser formulados naturalmente em termos de razões. Por exemplo, em estratégias financeiras como a estratégia de média de custo, um valor fixo é gasto em ações de moedas em intervalos regulares. Quanto maior o preço da moeda, então, menor o numero de ações que um investidor seguindo esta estratégia compra. O valor médio (média aritmética) que este investidor paga pelas ações é a média harmônica dos preços.

## As estimativas pontuais podem ser enganosas

As médias, por natureza, escondem muita informação, pois colapsam distribuições inteiras em um número. Como resultado, muitas vezes "estimativas pontuais" ou métricas que usam um número, podem disfarçar caracteristicas importantes em seus dados. Você deve ter cuidado para garantir que não esteja perdendo informações importantes ao resumir seus dados, e raramente deve usar uma média sem se referir a uma medida de dispersão.

## A distribuição subjacente pode estar errada

Mesmo quando você está usando as métricas certas para média e dispersão, elas podem não fazer sentido se sua distribuição subjacente não for o que você acha que é. Por exemplo, usar o desvio padrão para medir a freqüência de um evento geralmente assumirá a normalidade. Tente não assumir distribuições a menos que seja necessário, caso em que você deve verificar rigorosamente que os dados se encaixam na distribuição que você está assumindo.

# Desvio e Variância, Medidas de Dispersão

Dispersão mede o "espalhamento" de um conjunto de dados. Dados com baixa dispersão estão fortemente agrupados em torno da média, enquanto a alta dispersão a indica muitos valores muito afastados da média.

Para exemplificar, geramos uma série de numeros aléatórios com `numpy`.


```python
np.random.seed(42) # fixa a random seed
# Generate 20 random integers < 100
X = np.random.randint(100, size=20)

# Sort them
X = np.sort(X)
print('X: %s' %(X))

mu = np.mean(X)
print('Média de X:', mu)
```

    X: [ 1  2 14 20 21 23 29 37 51 52 60 71 74 74 82 86 87 87 92 99]
    Média de X: 53.1


# Amplitude

Amplitude é simplesmente a diferença entre os valores máximo e mínimo em um conjunto de dados. Não surpreendentemente, é muito sensível a *outliers*. Usaremos a função `numpy` pico a pico (`ptp`) para mensurar esta estimativa.


```python
print('Amplitude ptp de X: %s' %(np.ptp(X)))
```

    Amplitude ptp de X: 98


# Desvio Absoluto Médio (MAD)

O desvio absoluto médio é a média das distâncias das observações da média aritmética. Usamos o valor absoluto do desvio, de modo que 5 acima da média e 5 abaixo da média, ambos contribuem 5, pois, de outra forma, os desvios sempre somam para 0.

$$ MAD = \frac {\sum_ {i = 1} ^ n | X_i - \mu |} {n} $$


onde $ n $ é o número de observações e $ \mu $ é o sua média.


```python
abs_dispersion = [np.abs(mu - x) for x in X]
MAD = np.sum(abs_dispersion)/len(abs_dispersion)
print('Desvio absoluto médio de X:', MAD)
```

    Desvio absoluto médio de X: 28.099999999999994


# Variação e Desvio padrão

A variância $ \sigma ^ 2 $ é definida como a média dos desvios quadrados em torno da média:
$$ \sigma ^ 2 = \frac {\sum_ {i = 1} ^ n (X_i - \mu) ^ 2} {n} $$

Isso às vezes é mais conveniente do que o desvio absoluto médio porque o valor absoluto não é diferenciável, enquanto seu quadrado é suave e alguns algoritmos de otimização dependem da diferenciabilidade.

O desvio padrão é definido como a raiz quadrada da variância, $ \sigma $, e é de mais fácil interpretação pois está nas mesmas unidades das observações.


```python
print('Variancia de X:', np.var(X))
print('Desvio padrão de X:', np.std(X))
```

    Variancia de X: 990.49
    Desvio padrão de X: 31.472051092993606


Uma maneira de interpretar o desvio padrão é se referindo à desigualdade de Chebyshev. Isso nos diz que a proporção de amostras dentro dos desvios-padrão de $ k $ (isto é, dentro de uma distância de $ k \cdot $ desvio padrão) da média é pelo menos $ 1 - 1 /k ^ 2 $ para todos $ k> 1 $.

Vamos verificar se isso é verdade para o nosso conjunto de dados.


```python
k = 1.25
dist = k*np.std(X)
l = [x for x in X if abs(x - mu) <= dist]
print('Observações dentro de', k, 'stds da média:', l)
print('Confirmando que', float(len(l))/len(X), '>', 1 - 1/k**2)
```

    Observações dentro de 1.25 stds da média: [14, 20, 21, 23, 29, 37, 51, 52, 60, 71, 74, 74, 82, 86, 87, 87, 92]
    Confirmando que 0.85 > 0.36


O limite dado pela desigualdade de Chebyshev parece bastante "largo" neste caso. Este limite raramente é rigoroso, mas é útil porque é válido para todos os conjuntos de dados e distribuições.

# Semi Variância e Semi Desvio padrão

Embora variância e desvio padrão nos digam o quanto uma quantidade é volátil, eles não diferenciam os desvios para cima e os desvios para baixo. Muitas vezes, como no caso de retornos sobre um ativo, estamos mais preocupados com os desvios para baixo. Isto é abordado pela semi variância e pelo semi desvio padrão, que contam apenas as observações que se encontram abaixo da média. A semi variância é definida como
$$ \frac {\sum_ {X_i <\mu} (X_i - \mu) ^ 2} {n_<} $$
onde $ n_<$ é o número de observações que são menores do que a média. O semi desvio padrão é a raiz quadrada da semi variância.

Uma noção relacionada é semi variância alvo (e semi desvio padrão alvo), onde medimos a distância entre os valores que se situam abaixo desse alvo:

$$ \frac {\sum_ {X_i <B} (X_i - B) ^ 2} {n_{<B}} $$


```python
B = 19
lows_B = [e for e in X if e <= B]
semivar_B = sum(map(lambda x: (x - B)**2,lows_B))/len(lows_B)

print('Semi variancia alvo de X:', semivar_B)
print ('Semi desvio padrão alvo de X:', np.sqrt(semivar_B))
```

    Semi variancia alvo de X: 212.66666666666666
    Semi desvio padrão alvo de X: 14.583095236151571


## Estas são apenas estimativas

Todos esses cálculos fornecerão estatísticas de amostra, como o desvio padrão de uma amostra de dados. Se isso reflete ou não o desvio padrão real da população, nem sempre é óbvio, e é necessário análise minuciosa para determinar sua validade. 
