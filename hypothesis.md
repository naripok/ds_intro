# Teste de Hipotese, p-hacking e Viés de Comparações Múltiplas  

A inferência estatística, a prática de fazer previsões sobre um grande grupo com base em amostras menores, é tradicionalmente dividida em dois segmentos, ** estimativa** e **teste de hipóteses**. Estimativa fornece valores para medidas específicas as quais podme ser de  interesse, como média ou variância, com um intervalo de confiança fornecido. Um intervalo de confiança fornece uma região na qual você pode esperar encontrar o verdadeiro valor do parâmetro estimado, já que uma estimativa nunca será exata. Os intervalos de confiança usam um nível de confiança definido para escolher a largura do intervalo, para obter uma maior confiança, devemos informar um intervalo mais amplo. Para mais informações, consulte o artigo [Intervalos de Confiança] (./confidence_intervals.html).

Por exemplo, podemos estimar a média de uma amostra sendo $ 100 $, com um intervalo de confiança de $ 90, 110 $ a um nível de confiança $ 95 \% $. Isso não significa que a média real da população esteja entre $ 90 $ e $ 110 $ com $ 95 \% $ probabilidade, pois a verdadeira média é um valor fixo e a probabilidade para o valor é $ 100 \% $ ou $ 0 \% $, porém não sabemos este valor. Ao invés disso, o que isto significa é que, ao longo de muitos cálculos de um intervalo de confiança de $ 95 \% $, assumindo que pressupostos subjacentes sobre as distribuições sejam mantidos, a média da população estará no intervalo $ 95 \% $ do tempo.

Isso nos dá uma idéia das características específicas que uma população pode exibir, dada uma amostra. O teste de hipóteses fornece um foco diferente, detalhando uma estrutura para testes estatísticos para valores hipotéticos. Ao fazer uma afirmação sobre que valor certa medida deve apresentar, criamos uma hipótese testável.

Uma coisa a ter em mente é que os testes estatísticos são projetados de tal forma que, se todas as condições pré-requisitos forem verdadeiras, você deve obter a resposta correta sobre os dados uma determinada porcentagem das vezes. Quando você aceita uma hipótese como verdadeira com base em um teste, isso não significa que ela seja definitivamente verdadeira. Significa apenas que você sabe a probabilidade de estar errado.

## A hipótese nula e a hipótese alternativa

Inicialmente definimos a hipótese nula, normalmente nomeada $ H_0 $. A hipótese nula representa o caso padrão, geralmente refletindo o entendimento geral sobre a idéia. A hipótese alternativa é aquela a qual você testa.


## Exemplos

A hipótese alternativa $ H_A $ é que você possui mais do que 10 pares de sapatos.
A hipótese nula $ H_0 $ é que você não possui mais do que 10 pares de sapatos.

A hipótese alternativa $ H_A $ é que a Bitcoin apresenta retornos positivos.
A hipótese nula $H_0$ é de que a Bitcoin não apresenta retornos positivos.

## Dificuldades na realização dos testes

Algumas hipóteses são mais fáceis de testar do que outras. A hipótese de que você possui mais do que 10 pares de sapatos e sua hipótese alternativa são facilmente testadas, apenas contando-se o número de pares de sapatos que você possui, e esta apresentará uma incerteza bem pequena, pois a chance de um erro na contagem é bem baixa.  
  
O mesmo não pode ser dito do segundo par de hipóteses nula e alternativa, sobre os retornos do Bitcoin (acredite, eu tentei), e um grande número de amostras deverão ser tomadas antes de se poder afirmar algo com alguma certeza.

## As hipóteses devem ser "testáveis"

As hipóteses não podem ser "vagas", de forma que não possam ser testadas. Por exemplo, "*Daytrading* é uma boa maneira de fazer dinheiro" não é tão claro. O que "boa" quer dizer nesta frase (provavelmente que a pessoa não sabe nada sobre estatística)? As hipóteses devem ser claras e objetivas, e o teste deve ser facilmente derivável destas.

## Execução do teste de hipótese

Os passos para se executar um teste de hipótese são os seguintes:

1. Definir as hipóteses nula e alternativa.
2. Identificar o teste e a distribuição subjacente apropriada. Assegurar de que todas as premissas sobre os dados sejam válidas (estacionariedade, normalidade, etc).
3. Escolher um nível de significância, $ \alpha $
4. De $ \alpha $ de da distribuição, calcular o 'valor crítico'
5. Coletar dados e calcular o teste estatístico
6. Comparar o valor do teste com o valor crítico e decidir sobre aceitar ou rejeitar a hipótese.


Primeiro definine-se a hipótese que se quer testar. faz-se isso identificando-se a **hipótese nula**, $ H_0 $, e a **hipótese alternativa**, $ H_A $., sendo a hipótese nula aquela a qual se quer testar e a hipótese alternativa aquela a ser aceita caso $ H_0 $ for rejeitada.

Digamos que gostaríamos de testar o caso do retorno das ações da microsoft serem positivos. O parâmetro que estariamos testando é definido como $\theta$ e o valor proposto para este parâmetro é definido como $\theta_0$, o quao, neste caso, tem valor $0$. Então diríamos que nossa $H_0$ é de que $\theta = \theta_0$, sendo os retornos negativos, e nossa $H_A$ é de que  $\theta \neq \theta_0$. Levando-se em conta esta formulação, existem três possíveis maneiras de se formular as hipóteses nula e alternativa:

1. $H_0: \theta = \theta_0$ versus $H_A: \theta \neq \theta_0$ (Uma hipótese alternativa de "diferente de")
2. $H_0: \theta \leq \theta_0$ versus $H_A: \theta > \theta_0$ (Uma hipótese alternativa de "maior que")
3. $H_0: \theta \geq \theta_0$ versus $H_A: \theta < \theta_0$ (Uma hipótese alternativa de "menor que")

Neste caso, estaríamos testando os retornos da ação da Microsoft (MSFT), com $\theta = \mu_{MSFT}$, representando o retorno médio desta ação.
Como estaríamos testando os casos dos retornos serem positivos ou negativos, temos que$\theta_0 = 0$. Este exemplo segue a primeira formulação do teste de hipótese. Este é denominado **teste de hipótese bilateral**. As duas formulações subsequentes são chamadas de **testes de hipótese unilaterais**. Com o teste unilateral, rejeitamos a hipótese nula em favor da alternativa apenas no caso dos dados indicarem que $\theta$ é, respectivamente, maior que ou menor que $\theta_0$. O teste de hipótese bilateral rejeita $H_0$ caso os dados indiquem tanto que $\theta$ é maior $\theta_0$, quanto menor que este.
  
Então, caso quisermos descrever o teste de hipótese sobre os retornos da MSFT em termos qualitativos, diríamos:

So if we were to write out our hypothesis for MSFT in more qualitative terms, we would have:

\begin{eqnarray}
H_0 &:& \text{Que o retorno das ações da Mirosoft é $0$}\\
H_A &:& \text{Que o retorno das ações da Mirosoft não é $0$}
\end{eqnarray}

Quando definimos um teste de hipótese, a hipótese nula e a hipótese alternativa devem necessariamente serem complementares. Entre elas devem estar cobertos todos os possíveis valores de $\theta$. Independente do tipo de teste que quisermos realizar, sempre devemos testar a hipótese nula como se $\theta = \theta_0$. Mesmo no caso em que o teste é unilateral, isto deverá nos fornecer evidências suficientes para tomarmos uma decisão. Por exemplo, no caso de $H_0: \theta \leq 0$, $H_A: \theta > 0$, e de termos evidência suficiente para rejeitar $H_0: \theta = 0$ em favor de $H_A: \theta > 0$, então isto deve se manter verdadeiro para todos os valores menores que $0$.

O caso mais comum de teste de hipótese é o bilateral, "diferente de", pois este representa a premissa imparcial. Os testes de hipótese unilaterais são parciais, e normalmente utilizados quando se tem algum conhecimento *a priori* sobre o assunto, para testar por valores esperados.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
%matplotlib inline
```

Vamos aplicar estes testes sobre alguns dados.


```python
msft = web.DataReader('MSFT', 'robinhood').close_price.apply(float)

#transform it into returns
returns_sample = msft.pct_change()[1:]

# plot it
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.plot(msft.index.levels[1], msft)
plt.ylabel('Preço');
```


![png](hypothesis_files/hypothesis_5_0.png)



```python
# plot it
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.plot(msft.index.levels[1][1:], returns_sample)
plt.ylabel('Retornos');
```


![png](hypothesis_files/hypothesis_6_0.png)


## Por que tudo isso é necessário?

Por que não podemos simplesmente obter os retornos médios da microsoft e verificar se eles são > 0? Como não podemos analisar o processo real de geração de dados por trás dos retornos, podemos apenas amostrar retornos em um período de tempo limitado. Como só observamos uma amostra, essa amostra pode ou não refletir o verdadeiro estado do processo subjacente. Por causa dessa incerteza, precisamos usar testes estatísticos.

A seguir, identificamos o teste estatístico apropriado e sua distribuição subjacente. O teste estatístico geralmente tem a forma:

$$ \text{Estatística em teste} = \frac{\text{Estatística da amostra} - \text{Valor do parâmetro da população sobre $H_0$ ($\theta_0$)}}{\text{Erro padrão da estatística da amostra}} $$
  
A estatística em teste é calculada com base nos dados da amostra e é comparada com a sua distribuição de probabilidade para se determinar a rejeição ou nao rejeição da hipótese nula. Como estamos realizando um teste sobre o valor médio dos retornos da Microsoft, podemos utilizar uma média simples,  $\bar{X}_\mu$, como nossa estatística da amostra. Calculamos o erro padrão da média da amostra como $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$ se soubermos o desvio padrão desta medida, $\sigma$, ou como $s_{\bar{X}} = \frac{s}{\sqrt{n}}$, onde $s$ é o desvio padrão da amostra. Então, usando estas definições, nosso teste estatístico é calculado como:

$$ \frac{\bar{X}_\mu - \theta_0}{s_{\bar{X}}} = \frac{\bar{X}_\mu - 0}{s/\sqrt{n}} $$
  
As quatro distribuições mais comuns para testes estatísticos são as seguintes:

* A distribuição $t$ (teste $t$)
* A distribuição normal (teste $z$)
* A distribuição chi-quadrada (teste $\chi^2$)
* A distribuição $F$ (teste $F$)


Vamos cobrir estas em detalhes mais tarde. Agora, assumiremos que podemos utilizar o teste $z$ com suas premissas em nosso exemplo com a Microsoft.


Depois de identificarmos o teste apropriado e sua distribuição de probabilidade, precisamos especificar o nível de significância para o teste, $\alpha$. Os valores com os quais compararemos a estatística do teste para rejeitar ou falhar em rejeitar a hipótese nula são determinados com base no nível de significância.

|| Situação Real ||
| :---: | :---: | :---: |
| **Decisão** | $H_0$ verdadeira | $H_0$ falsa |
| Não rejeita $H_0$ | Decisão correta | Erro tipo II |
| Rejeita $H_0$ (aceita $H_A$) | Erro tipo I | Decisão correta |

O nível de significância é igual a probabilidade de um erro do tipo I (falso positivo) ocorrer. A probabilidade de um erro do tipo II (falso negativo) ocorrer é denominada $\beta$. Se quisermos diminuir a probabilidade de um erro do tipo I ocorrer, automaticamente aumentamos a probabilidade de um erro do tipo II, resultando em uma troca. A única forma de reduzir-se ambas as probabilidades de erro do tipo I e tipo II ocorrerem é aumentando o tamanho da amostra.

Os níveis de significância mais utilizados são $0.1$, $0.05$, and $0.01$. Rejeitar a hipótese nula com nível de significância $0.1$ quer dizer que temos alguma evidência de que $H_0$ é falsa, já para $\alpha = 0.05$ tem-se forte evidência de que $H_0$ é falsa e para $0.01$ tem-se evidência fortíssima de que $H_0$ é falsa.

### Valor crítico

Agora descobrimos nosso valor crítico, ou ponto de rejeição. O valor crítico para nosso teste estatístico é o valor ao qual comparamos a estatística em teste quando decidimos rejeitar a hipótese nula. Se rejeitarmos $H_0$, dizemos que o resultado é **estatisticamente significante**, enquanto se não rejeitarmos $H_0$, dizemos que o resultado é **não estatisticamente significativo**.

Nós comparamos nossa estatística de teste com um **valor crítico**, a fim de decidir se rejeitamos ou não a hipótese nula. O valor crítico de um teste é determinado com base no $ \alpha $ do nosso teste de hipótese, bem como na distribuição escolhida. No nosso caso, digamos que $ \alpha = 0.05 $, então nosso nível de significância é $ 0.05 $. Com um teste $ z $ unilateral, existem duas maneiras diferentes de ver os valores críticos:

* Se testamos $H_0$: $\theta \leq \theta_0$, $H_A$: $\theta > \theta_0$ at $\alpha = 0.05$, nosso valor crítico é $z_{0.05} = 1.645$. Então comparamos nossa estatística de teste e rejeitamos $H_0$ caso $z > 1.645$.
* Se testamos $H_0$: $\theta \geq \theta_0$, $H_A$: $\theta < \theta_0$ at $\alpha = 0.05$, nosso valor crítico é $-z_{0.05} = -1.645$. ntão comparamos nossa estatística de teste e rejeitamos $H_0$ caso $z < -1.645$.

Um teste bilateral é uma situação ligeiramente diferente. Como é de dois lados, existem dois pontos de rejeição, negativos e positivos. Nosso $ \alpha $ é $ 0,05 $, então a probabilidade de um erro do Tipo I deve totalizar $ 0,05 $. Dessa forma, dividimos $ 0,05 $ pela metade para que nossos dois pontos de rejeição sejam $ z_ {0,025} $ e $ -z_ {0,025} $ para os valores críticos positivo e negativo, respectivamente. Para um teste $ z $, esses valores são $ 1.96 $ e $ -1.96 $. Assim, rejeitamos $H_0$ se $ z <-1.96 $ ou se $ z> 1.96 $. Se acharmos que $ -1.96 \leq z \leq 1.96 $, nós falhamos em rejeitar $H_0$.

Ao realizar um teste de hipótese, você também pode usar um **valor - $ p $** para determinar o resultado. Um valor $ p $ é o nível mínimo de significância onde você pode rejeitar a hipótese nula. Muitas vezes as pessoas interpretarão os valores de $ p $ como a "probabilidade de que a hipótese nula seja falsa", mas isso é errôneo. Um valor $ p $ só faz sentido quando comparado com o valor de significância. Se um valor - $ p $ for menor que $ \alpha $, rejeitamos $H_0$ e, caso contrário, não o faremos. Valores menores de $ p $ não fazem algo "mais estatisticamente significativo". Muitos resultados estatísticos irão calcular um valor $ p $ para você, mas também é possível calculá-lo manualmente. O cálculo depende tanto do seu tipo de teste de hipótese quanto do CDF (coberto em [variáveis aleatórias] (./random_variables.html)) da distribuição com a qual você está trabalhando. Para calcular manualmente um valor $ p $, faça o seguinte:

* Em um teste do tipo "menor ou igual a", o valor - $p$ é $1 - CDF(\text{estatística do teste})$
* Em um teste do tipo "maior ou igual a", o valor - $p$ é $CDF(\text{estatística do teste})$
* Em um teste do tipo "diferente de", o valor - $p$ é $2 * 1 - CDF(\text{estatística do teste}|)$

Os valores de significância se encaixam muito bem nos intervalos de confiança, que são abordados mais detalhadamente em  [intervalos de confiança] (./confidence_intervals.html). Um intervalo de confiança nos fornece uma estimativa para o intervalo possível de um parâmetro em valores, dado um determinado nível de significância. Por exemplo, se nosso intervalo de confiança de $ 99 \% $ para a média dos retornos da MSFT for $ (- 0,0020, 0,0023) $, isso significaria que havia uma probabilidade de $ 99 \% $ de que o valor real da média estivesse dentro desse intervalo .


```python
# Plot a standard normal distribution and mark the critical regions with shading
x = np.linspace(-3, 3, 100)
norm_pdf = lambda x: (1/np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)
y = norm_pdf(x)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14,6))
ax.plot(x, y)
ax.fill_between(x, 0, y, where = x > 1.96)
ax.fill_between(x, 0, y, where = x < -1.96)
plt.title('região de rejeição para um teste bilateral com nível de confiança de 95%')
plt.xlabel('x')
plt.ylabel('p(x)');
```


![png](hypothesis_files/hypothesis_11_0.png)


Agora coletamos os dados relevantes para nosso teste e calculamos a estatística de teste para um teste de significância de bilateral, US $ 5 \% $. Tenha em mente que quaisquer características negativas dos dados afetarão negativamente o nosso teste de hipótese e possivelmente o tornarão inválido. No caso do nosso teste de retornos da MSFT, podemos ter problemas de viés de período de tempo ou de viés de antecipação (se prepararmos o teste incorretamente). Como sempre com dados históricos, os dados com os quais trabalhamos podem resultar em um resultado de teste específico que pode não ser válido para o futuro. Também temos que nos certificar de que os dados não incluam quaisquer valores que não conheceríamos durante o período de tempo que estamos testando (embora isso seja um problema maior ao se comparar vários valores com testes de hipóteses).

Aqui nós calculamos a estatística de teste:


```python
n = len(returns_sample)
test_statistic = ((returns_sample.mean() - 0) /
                (returns_sample.std()/np.sqrt(n)))
print('estatística de teste t: ', test_statistic)
```

    estatística de teste t:  1.759480190968843


Para fazer a decisão estatística para o teste, comparamos nossa estatística de teste com nosso valor crítico. Nossa estatística de teste, como afirmado acima, está entre os dois valores críticos para um teste $ 95 $ bilateral de 95%. Neste exemplo, **não rejeitamos** nosso $ H_0 $, nossa hipótese de que os retornos da MSFT **não são** $ 0 $.

Se, em vez disso, optássemos por determinar o resultado desse teste de hipótese com um valor $ p $, calcularíamos o valor $ p $ da seguinte maneira:


```python
from scipy.stats import t
```


```python
p_val = 2 * (1 - t.cdf(test_statistic, n - 1))
print ('Valor-p : ', p_val)
```

    Valor-p :  0.07973398485192984


Como o valor $ p $ é maior que o nosso nível de significância, $ \alpha = 0,05 $, nós **não rejeitamos** a hipótese nula.

Depois que tomamos a decisão estatística, precisamos traduzi-la para o mundo real. Muitas vezes isso pode ser difícil de se fazer diretamente, mas os resultados podem ter outras implicações. No caso do nosso exemplo, descobrimos que os retornos diários da Microsoft em 2017 não eram significativamente diferentes de $ 0 $.

## Teste de hipóteses sobre médias

Uma distribuição $ z $, ou uma distribuição normal padrão, é uma distribuição de probabilidade essencial em estatística. Preferimos quando as coisas são normalmente distribuídas porque esta distribuição possui muitas propriedades úteis. Além disso, muitos métodos fundamentais exigem suposição de normalidade. No entanto, na maioria dos casos, uma distribuição $ z $ será inadequada para nossos dados. Raramente sabemos os valores reais dos parâmetros (média e variância) de nossos dados e devemos confiar em aproximações. Nesses casos, devemos usar a distribuição $ t $ e a aproximação da distribuição normal. A distribuição $ t $ é mais tolerante quando se trata de pequenos tamanhos de amostra e deve ser usada com a média e a variância da amostra. Tem cauda mais grossa e um pico mais baixo, dando mais flexibilidade em comparação com uma distribuição normal.

As distribuições $ t $ e $ z $ dependem de uma suposição subjacente de normalidade. Como tal, além de testar médias individuais, faz sentido usá-los para comparar entre dois ou mais valores médios. Podemos usar um teste de hipótese para determinar se as médias de vários conjuntos de dados são estatisticamente diferentes umas das outras. Aqui, usaremos uma distribuição $ t $ para demonstrar. Vamos comparar os retornos médios das ações da S & P500 e da Apple com um teste de hipótese para ver se as diferenças são estatisticamente significativas ou não.


```python
symbols = ['AAPL', 'SPY']
stocks = {}
for symbol in symbols:
    stocks[symbol] = web.DataReader(symbol, 'robinhood').close_price.apply(float)
    stocks[symbol].index = stocks[symbol].index.levels[1]
        
stocks = pd.DataFrame.from_dict(stocks)
returns = stocks.pct_change()[1:]

plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
for symbol in returns:
    plt.plot(returns[symbol])
plt.ylabel('Retornos');
```


![png](hypothesis_files/hypothesis_21_0.png)


Ainda que estas sequências aparentem ter a mesma média, ainda não temos evidências suficientes para constatar. Usamos um teste de hipótese para dar base estatística a nossas suspeitas.

Quando comparando duas médias, os testes de hipóteses são formulados como:

1. $H_0: \mu_1 - \mu_2 = \theta_0, \ H_A: \mu_1 - \mu_2 \neq \theta_0$
2. $H_0: \mu_1 - \mu_2 \leq \theta_0, \ H_A: \mu_1 - \mu_2 > \theta_0$
3. $H_0: \mu_1 - \mu_2 \geq \theta_0, \ H_A: \mu_1 - \mu_2 < \theta_0$

Onde $ \mu_1, \mu_2 $ são as respectivas médias de SPY e AAPL e $ \theta_0 $ é o parâmetro que estamos testando. Usaremos o primeiro teste de hipótese para testar a igualdade dos dois retornos. Se assumirmos que as variâncias populacionais são iguais, nossa estatística de teste é calculada como:

$$ t = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{(\frac{s_p^2}{n_1} + \frac{s_p^2}{n_2})^{1/2}} = \frac{\bar{X}_1 - \bar{X}_2}{(\frac{s_p^2}{n_1} + \frac{s_p^2}{n_2})^{1/2}}$$

Com $s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2)}{n_1 + n_2 - 2}$ como o estimador da variância comum, conhecido como a variação combinada, e $ n_1 + n_2 - 2 $ como o número de graus de liberdade ($ n_1 - 1 $ e $ n_2 - 1 $ para cada conjunto de dados). Um teste típico de $ t $ numa média pressupõe que todas as variâncias envolvidas são iguais e com distribuições normais. Se estamos assumindo que as variâncias não são iguais, temos que calcular nossa estatística de teste de maneira diferente. Nossa estatística de teste neste caso é:

$$ t = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^{1/2}} = \frac{\bar{X}_1 - \bar{X}_2}{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^{1/2}}$$

Onde o número de graus de liberdade usados para encontrar a estatística crítica é dos graus de liberdade dos parâmetros, o número de valores que estão livres para variar, $ df = \ frac {(\ frac {s_1 ^ 2} {n_1} + \ frac {s_2 ^ 2} {n_2}) ^ 2} {\ frac {(s_1 ^ 2 / n_1) ^ 2} {n_1} + \ frac {(s_2 ^ 2 / n_2) ^ 2} {n_2}} $. Isso preserva a normalidade subjacente dos dados que estão sendo testados enquanto contabiliza variâncias diferentes. Calcular a estatística dessa maneira elimina muitos problemas que podem ocorrer se tivermos variações desiguais, especialmente se os tamanhos das amostras dos dados subjacentes também diferirem. Este caso específico de um teste $ t $ é chamado de ["Teste-$t$ de Welch $ t $ para variâncias desiguais"] (https://en.wikipedia.org/wiki/Welch%27s_t-test).

Para este exemplo, estamos assumindo que as variâncias dos retornos SPY e AAPL são diferentes. Achamos que a AAPL será mais arriscada do que a SPY, por isso usaremos a segunda formulação da estatística de teste. Digamos que $ \alpha = 0.05 $, portanto, estamos computando um teste de hipótese de $ 95 \% $.


```python
# Sample mean values
mu_spy, mu_aapl = returns.mean()
s_spy, s_aapl = returns.std()
n_spy = len(returns['SPY'])
n_aapl = len(returns['AAPL'])

test_statistic = ((mu_spy - mu_aapl) - 0)/((s_spy**2/n_spy) + (s_aapl**2/n_aapl))**0.5
df = ((s_spy**2/n_spy) + (s_aapl**2/n_aapl))**2/(((s_spy**2 / n_spy)**2 /n_spy)+((s_aapl**2 / n_aapl)**2/n_aapl))

print('Estatística de teste-t: ', test_statistic)
print('Graus de liberdade: ', df)
```

    Estatística de teste-t:  0.3030010620329521
    Graus de liberdade:  384.3173644822406


Olhando para a [tabela-t] (https://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values), determinamos que os valores críticos para nosso teste de hipóteses bilateral são $ -1,96 $ e $ 1,96 $. Nossa estatística de teste está entre esses valores, portanto, **não rejeitamos** a hipótese nula e determinamos que a diferença entre os retornos de SPY e AAPL é **não** significativamente diferente de $ 0 $.

## Teste de hipóteses sobre desvios

Se quisermos testar as variâncias das populações, precisamos usar uma distribuição diferente das distribuições $ t $ e $ z $. As variações devem, por definição, ser maiores que (ou iguais a) $ 0 $ e o fato de as distribuições com as quais trabalhamos até agora permitirem valores negativos as inviabiliza como distribuições de teste. O risco é quantificado em termos de desvios padrão e variâncias, portanto, esse método de teste de hipóteses é uma adição útil à nossa caixa de ferramentas.

Em vez das distribuições $ t $ e $ z $, trabalharemos com distribuições $ \chi ^ 2 $ para testes de variância única e distribuições $ F $ para comparações de variância. Essas distribuições são limitadas abaixo por $ 0 $, tornando-as viáveis para testes desse tipo.

Assim como com todos os nossos outros testes de hipóteses, os testes de uma única variação podem assumir três formas:

1. $H_0: \sigma^2 = \sigma_0^2, \ H_A: \sigma^2 \neq \sigma_0^2$
2. $H_0: \sigma^2 \leq \sigma_0^2, \ H_A: \sigma^2 > \sigma_0^2$
3. $H_0: \sigma^2 \geq \sigma_0^2, \ H_A: \sigma^2 < \sigma_0^2$

A distribuição $ \chi ^ 2 $ é uma família de funções com cada formulação diferente determinada pelo número de graus de liberdade. A forma da distribuição é diferente para cada valor diferente do número de graus de liberdade, $ k $.


```python
from scipy.stats import chi2
```


```python
# Here we show what a chi-square looks like
x = np.linspace(0, 8, 100)
y_1 = chi2.pdf(x, 1)
y_2 = chi2.pdf(x, 2)
y_3 = chi2.pdf(x, 3)
y_4 = chi2.pdf(x, 4)
y_6 = chi2.pdf(x, 6)
y_9 = chi2.pdf(x, 9)

fig, ax = plt.subplots(figsize=(14,6))
plt.grid()
ax.plot(x, y_1, label = 'k = 1')
ax.plot(x, y_2, label = 'k = 2')
ax.plot(x, y_3, label = 'k = 3')
ax.plot(x, y_4, label = 'k = 4')
ax.plot(x, y_6, label = 'k = 6')
ax.plot(x, y_9, label = 'k = 9')
ax.legend()
plt.title('Distribuiçõ Chi-quadrado com k graus de liberdade')
plt.xlabel('x')
plt.ylabel('p(x)');
```


![png](hypothesis_files/hypothesis_27_0.png)


Calculamos a estatística de teste com a  $\chi^2$ como:

$$ \chi^2 = \frac{(n - 1)s^2}{\sigma_0^2} $$

Onde $s^2$ é a variância da amostra e $n$ é o tamanho da mesma. O número de graus de liberdade é $n - 1$ e é utilizado em conjunto com a estatística de teste para determinar o valor crítico do teste de hipótese $\chi^2$.


```python
# plot it
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.plot(msft.index.levels[1][1:], returns_sample)
plt.ylabel('Retornos');
```


![png](hypothesis_files/hypothesis_29_0.png)


Agora, usaremos um teste $ \chi^2 $ para testar o valor da variação das ações da Microsoft. Digamos que queremos usar $ \alpha = 0,01 $ para testar se a variância da MSFT é menor ou igual a $ 0,0001 $ (que o desvio padrão, ou risco, é menor ou igual a $ 0,01 $).
  
$$ H_0: \sigma^2 \leq 0.0001, \ H_A: \sigma^2 > 0.0001 $$

Para isso, calculamos a estatística de teste:


```python
test_statistic = (len(returns_sample) - 1) * returns_sample.std()**2 / 0.0001
print('Estatística de teste Chi-quadrado: ', test_statistic)
```

    Estatística de teste Chi-quadrado:  351.48869789240626



```python
# Here we calculate the critical value directly because our df is too high for most chisquare tables
crit_value = chi2.ppf(0.99, len(returns_sample) - 1)
print ('Valor crítico para a = 0.01 com %d df: ' %  returns.shape[0], crit_value)
```

    Valor crítico para a = 0.01 com 248 df:  301.6263657123845


Como estamos usando a formulação 'menor ou igual que' de um teste de hipótese unilateral, rejeitamos a hipótese nula se nossa estatística de teste for maior que o valor crítico. Como $ 351.491 > 302.730 $, nós **rejeitamos** a hipótese nula em favor da alternativa e afirmamos que $ \sigma ^ 2> 0.0001 $.

### Comparando duas variâncias

Podemos comparar as variações de duas distribuições separadas usando a distribuição $ F $. Ao construir uma comparação de
variâncias usando um teste $ F $, as formulações de hipóteses são as mesmas:

1. $H_0: \sigma_1^2 = \sigma_2^2, \ H_A: \sigma_1^2 \neq \sigma_2^2$
2. $H_0: \sigma_1^2 \leq \sigma_2^2, \ H_A: \sigma_1^2 > \sigma_2^2$
3. $H_0: \sigma_1^2 \geq \sigma_2^2, \ H_A: \sigma_1^2 < \sigma_2^2$

A distribuição $ F $ é semelhante à distribuição $ \chi ^ 2 $, por ser assimétrica e limitada por $ 0 $. A distribuição $ F $ é definida com dois valores diferentes de graus de liberdade. Para fins de teste de hipóteses, cada um correlaciona-se com um dos fatores que estamos comparando. Uma distribuição $ F $ pode ser construída a partir de duas distribuições $ \chi ^ 2 $ separadas. $ X $ é uma variável aleatória $ F $ se puder ser escrita como $ X = \frac {Y_1 / d_1} {Y_2 / d_2} $, onde $ Y_1 $ e $ Y_2 $ são  variáveis aleatórias em $ \chi ^ 2 $ com graus de liberdade $ d_1 $ e $ d_2 $, respectivamente.

A variável aleatória $ F $ é essencialmente uma proporção de variações. Consequentemente, a construção da estatística de teste $ F $ é feita considerando a proporção das variâncias de amostra dos dados que queremos testar. Podemos simplesmente escolher $ \sigma_1 ^ 2 $ e $ \sigma_2 ^ 2 $ para representar um dos desvios que estamos comparando de forma que nossa estatística F seja maior que $ 1 $.

$$ F = \frac{s_1^2}{s_2^2} $$

Vamos comparar o SPY e o AAPL para ver se suas variâncias são as mesmas (um teste de hipóteses 'diferente de'). Nós usaremos um teste com nível de significância $ \alpha = 0.05 $. Lembre-se de que, para um teste bilateral, calculamos os valores críticos inferior e superior usando valores de $ \alpha / 2 $. Reunimos os dados e calculamos a estatística de teste.


```python
# Take returns from above, AAPL and SPY, and compare their variances
std = returns.std()
for symbol in std.index:
    print("Desvio padrão para %s: %f" % (symbol, std[symbol]))
```

    Desvio padrão para AAPL: 0.012649
    Desvio padrão para SPY: 0.006922


Note que o desvio padrão de AAPL é maior que o desvio padrão de SPY. Como resultado, escolhemos $ \sigma_1 ^ 2 $ para representar a variação de AAPL e $ \sigma_2 ^ 2 $ para representar a variação do SPY.


```python
test_statistic = (std['AAPL'] / std['SPY'])**2
print ("Estatística de teste F: ", test_statistic)
```

    Estatística de teste F:  3.3390847325128328



```python
for symbol in std.index:
    print("Graus de liberdade para %s: %d" % (symbol, returns[symbol].shape[0]))
```

    Graus de liberdade para AAPL: 248
    Graus de liberdade para SPY: 248



```python
from scipy.stats import f
```


```python
df1 = df2 = 249

upper_crit_value = f.ppf(0.975, df1, df2)
lower_crit_value = f.ppf(0.025, df1, df2)
print('Limite superior do valor crítico para a = 0.05 com df1 = {0} e df2 = {1}: '.format(df1, df2), upper_crit_value)
print('Limite inferior do valor crítico para a = 0.05 com df1 = {0} e df2 = {1}: '.format(df1, df2), lower_crit_value)
```

    Limite superior do valor crítico para a = 0.05 com df1 = 249 e df2 = 249:  1.2827228078241388
    Limite inferior do valor crítico para a = 0.05 com df1 = 249 e df2 = 249:  0.7795916576054985


Vemos que nosso valor de estatística F é maior que o valor crítico superior para nosso teste F. Assim, rejeitamos a hipótese nula em favor da alternativa e concluímos que as variâncias de AAPL e SPY realmente diferem.
