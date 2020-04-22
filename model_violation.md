# Violação das Suposições do Modelos

Ao usar uma regressão para ajustar um modelo a dados, os pressupostos da análise de regressão devem ser satisfeitos para garantir boas estimativas de parâmetros e estatísticas de ajuste precisas. Nós gostaríamos que os parâmetros fossem:
* imparcial (o valor esperado em diferentes amostras é o valor verdadeiro)
* consistente (convergente para o valor verdadeiro com muitas amostras) e
* eficiente (variância minimizada)

Abaixo investigamos as formas em que esses pressupostos podem ser violados e o efeito sobre os parâmetros e as estatísticas de ajuste. Usaremos regressões lineares de variável única para os exemplos, mas as mesmas considerações se aplicam a outros modelos. Também assumiremos que nosso modelo está corretamente especificado; ou seja, que a forma funcional que escolhemos é válida. Discutimos erros de especificação do modelo, juntamente com as violações da suposição e outros problemas que elas causam em outro notebook.

# Foco nos Residuais

Ao invés de se concentrar na construção do seu modelo, é possível obter uma grande quantidade de informações de seus resíduos (erros). Seu modelo pode ser incrivelmente complexo e impossível de analisar, mas desde que você tenha previsões e valores observados, você pode calcular os resíduos. Uma vez que você tenha seus resíduos, você pode realizar muitos testes estatísticos.

Se os seus resíduos não seguem uma dada distribuição (geralmente normal, mas depende do seu modelo), então você sabe que algo está errado e você deve se preocupar com a precisão de suas previsões.

# Residuais normalmente não distribuídos

Se o termo de erro não for normalmente distribuído, nossos testes de significância estatística darão respostas erroneas. Felizmente, o teorema do limite central nos diz que, para amostras de dados suficientemente grandes, as distribuições de coeficientes serão próximas a normal, mesmo que os erros não sejam. Portanto, nossa análise ainda será válida para grandes conjuntos de dados.

## Testando a normalidade

Uma boa prova de normalidade é o teste de Jarque-Bera. Tem uma implementação do python em `statsmodels.stats.stattools.jarque_bera`, usaremos com freqüência neste notebook.

### Teste sempre a normalidade!

É incrivelmente fácil e pode poupar um bom tempo.


```python
# Import all the libraries we'll be using
import numpy as np
import statsmodels.api as sm
from statsmodels import regression, stats
import statsmodels
import matplotlib.pyplot as plt
import pandas_datareader as web
%matplotlib inline
```

    /home/tau/Envs/ds/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```python
"""Jarque-Bera"""
residuals = np.random.normal(0, 1, 100)

_, pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(residuals)
print(pvalue)

residuals = np.random.poisson(size = 100)

_, pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(residuals)
print(pvalue)
```

    0.09669683787646297
    0.0030452463443241136


# Heterosqueticidade

Heterosqueticidade significa que a variância dos termos de erro não é constante em todas as observações. Intuitivamente, isso significa que as observações não estão uniformemente distribuídas ao longo da linha de regressão. Muitas vezes ocorre em dados transversais onde as diferenças nas amostras que estamos medindo levam a diferenças na variância.


```python
# Artificially create dataset with constant variance around a line
xs = np.arange(100)
y1 = xs + 3*np.random.randn(100)

# Get results of linear regression
slr1 = regression.linear_model.OLS(y1, sm.add_constant(xs)).fit()

# Construct the fit line
fit1 = slr1.params[0] + slr1.params[1]*xs

# Plot data and regression line
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.scatter(xs, y1)
plt.plot(xs, fit1)
plt.title('Errors homosquedasticos');
plt.legend(['Previsão', 'Observados'])
plt.xlabel('X')
plt.ylabel('Y');
```


![png](model_violation_files/model_violation_6_0.png)



```python
# Artificially create dataset with changing variance around a line
y2 = xs*(1 + .5*np.random.randn(100))

# Perform linear regression
slr2 = regression.linear_model.OLS(y2, sm.add_constant(xs)).fit()
fit2 = slr2.params[0] + slr2.params[1]*xs

# Plot data and regression line
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.scatter(xs, y2)
plt.plot(xs, fit2)
plt.title('Errors Heterosquedasticos');
plt.legend(['Previsão', 'Observados'])
plt.xlabel('X')
plt.ylabel('Y')

# Print summary of regression results
slr2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.499</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.494</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   97.53</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 25 Mar 2018</td> <th>  Prob (F-statistic):</th> <td>2.25e-16</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:37:09</td>     <th>  Log-Likelihood:    </th> <td> -482.83</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   969.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    98</td>      <th>  BIC:               </th> <td>   974.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    0.9817</td> <td>    6.065</td> <td>    0.162</td> <td> 0.872</td> <td>  -11.054</td> <td>   13.018</td>
</tr>
<tr>
  <th>x1</th>    <td>    1.0453</td> <td>    0.106</td> <td>    9.876</td> <td> 0.000</td> <td>    0.835</td> <td>    1.255</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.608</td> <th>  Durbin-Watson:     </th> <td>   2.245</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.272</td> <th>  Jarque-Bera (JB):  </th> <td>   1.977</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.272</td> <th>  Prob(JB):          </th> <td>   0.372</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.423</td> <th>  Cond. No.          </th> <td>    114.</td>
</tr>
</table>




![png](model_violation_files/model_violation_7_1.png)


### Testes para Heterosqueticidade

Você pode testar a heterocedasticidade usando alguns testes, usaremos o teste de *Breush Pagan* da biblioteca statsmodels. Também testaremos a normalidade, que neste caso também captura a não normalidade no segundo caso. No entanto, é possível ter resíduos normalmente distribuídos que também são heterocedasticos, então ambos os testes devem ser realizados para ter certeza.


```python
residuals1 = y1-fit1
residuals2 = y2-fit2

xs_with_constant = sm.add_constant(xs)

_, jb_pvalue1, _, _ = statsmodels.stats.stattools.jarque_bera(residuals1)
_, jb_pvalue2, _, _ = statsmodels.stats.stattools.jarque_bera(residuals2)
print ("Valor-p para residuals1 ser normal", jb_pvalue1)
print ("Valor-p para residuals2 ser normal", jb_pvalue2)

_, pvalue1, _, _ = stats.diagnostic.het_breuschpagan(residuals1, xs_with_constant)
_, pvalue2, _, _ = stats.diagnostic.het_breuschpagan(residuals2, xs_with_constant)
print ("Valor-p para residuals1 serem heterosquedásticos", pvalue1)
print ("Valor-p para residuals2 serem heterosquedásticos", pvalue2)
```

    Valor-p para residuals1 ser normal 0.7032136929431687
    Valor-p para residuals2 ser normal 0.37217482738134305
    Valor-p para residuals1 serem heterosquedásticos 0.27848322588674973
    Valor-p para residuals2 serem heterosquedásticos 3.1978455075349834e-08


### Corrigindo a heterosquedasticidade

Como a heterocedasticidade afeta nossa análise? A situação problemática, conhecida como heterocedasticidade condicional, é quando a variância do erro está correlacionada com as variáveis independentes, como está acima. Isso faz com que o teste F para significância de regressão e t-testes para as significâncias de coeficientes individuais sejam não confiáveis. Na maioria das vezes, isso resulta em uma superestimação da significância do ajuste.

O teste Breusch-Pagan e o teste White podem ser usados para detectar heterocedasticidade condicional. Se suspeitarmos que este efeito está presente, podemos alterar nosso modelo para tentar corrigi-lo. Um método é o mínimo de quadrados generalizados, o que requer uma alteração manual da equação original. Outro é o cálculo de erros padrão robustos, que corrige as estatísticas de ajuste para explicar a heterocedasticidade. `statsmodels` pode calcular erros padrão robustos; observe a diferença nas estatísticas abaixo.


```python
print(slr2.summary())
print(slr2.get_robustcov_results().summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.499
    Model:                            OLS   Adj. R-squared:                  0.494
    Method:                 Least Squares   F-statistic:                     97.53
    Date:                Sun, 25 Mar 2018   Prob (F-statistic):           2.25e-16
    Time:                        08:37:10   Log-Likelihood:                -482.83
    No. Observations:                 100   AIC:                             969.7
    Df Residuals:                      98   BIC:                             974.9
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.9817      6.065      0.162      0.872     -11.054      13.018
    x1             1.0453      0.106      9.876      0.000       0.835       1.255
    ==============================================================================
    Omnibus:                        2.608   Durbin-Watson:                   2.245
    Prob(Omnibus):                  0.272   Jarque-Bera (JB):                1.977
    Skew:                           0.272   Prob(JB):                        0.372
    Kurtosis:                       3.423   Cond. No.                         114.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.499
    Model:                            OLS   Adj. R-squared:                  0.494
    Method:                 Least Squares   F-statistic:                     90.34
    Date:                Sun, 25 Mar 2018   Prob (F-statistic):           1.44e-15
    Time:                        08:37:10   Log-Likelihood:                -482.83
    No. Observations:                 100   AIC:                             969.7
    Df Residuals:                      98   BIC:                             974.9
    Df Model:                           1                                         
    Covariance Type:                  HC1                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.9817      3.378      0.291      0.772      -5.721       7.685
    x1             1.0453      0.110      9.505      0.000       0.827       1.264
    ==============================================================================
    Omnibus:                        2.608   Durbin-Watson:                   2.245
    Prob(Omnibus):                  0.272   Jarque-Bera (JB):                1.977
    Skew:                           0.272   Prob(JB):                        0.372
    Kurtosis:                       3.423   Cond. No.                         114.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors are heteroscedasticity robust (HC1)


# Autocorrelação do erro

Um problema comum e grave é quando os erros estão correlacionados entre as observações (correlação serial ou autocorrelação). Isso pode ocorrer, por exemplo, quando alguns dos pontos de dados estão relacionados ou quando usamos dados de séries temporais com flutuações periódicas. Se uma das variáveis independentes depende dos valores anteriores da variável dependente - como, quando é igual ao valor da variável dependente no período anterior - ou se a especificação incorreta do modelo levar à autocorrelação, as estimativas dos coeficientes serão inconsistentes e portanto, inválidas. Do contrário, as estimativas dos parâmetros serão válidas, mas as estatísticas de ajuste serão imprecisas. Por exemplo, se a correlação for positiva, teremos inflação de estatísticas F e t, levando-nos a superestimar a significancia do modelo.

Se os erros são homosquedásticos, podemos testar a autocorrelação usando o teste Durbin-Watson, que é convenientemente relatado no resumo de regressão em `statsmodels`.


```python
# Load pricing data for an asset
y = web.DataReader('AAPL', 'robinhood').close_price.apply(float).values
x = np.arange(len(y))

# Regress pricing data against time
model = regression.linear_model.OLS(y, sm.add_constant(x)).fit()

# Construct the fit line
prediction = model.params[0] + model.params[1]*x

# Plot pricing data and regression line
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.plot(x,y)
plt.plot(x, prediction, color='r')
plt.legend(['DAL Price', 'Regression Line'])
plt.xlabel('Time')
plt.ylabel('Price')

# Print summary of regression results
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.794</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.793</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   953.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 25 Mar 2018</td> <th>  Prob (F-statistic):</th> <td>8.56e-87</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:37:11</td>     <th>  Log-Likelihood:    </th> <td> -777.43</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   249</td>      <th>  AIC:               </th> <td>   1559.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   247</td>      <th>  BIC:               </th> <td>   1566.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  140.2086</td> <td>    0.697</td> <td>  201.227</td> <td> 0.000</td> <td>  138.836</td> <td>  141.581</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.1501</td> <td>    0.005</td> <td>   30.886</td> <td> 0.000</td> <td>    0.141</td> <td>    0.160</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.674</td> <th>  Durbin-Watson:     </th> <td>   0.134</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.008</td> <th>  Jarque-Bera (JB):  </th> <td>  10.159</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.494</td> <th>  Prob(JB):          </th> <td> 0.00622</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.934</td> <th>  Cond. No.          </th> <td>    286.</td>
</tr>
</table>




![png](model_violation_files/model_violation_13_1.png)


### Testando Autocorrelação

Podemos testar a autocorrelação em ambos nossos preços e resíduos. Usaremos um método baseado no teste Ljun-Box. Este teste calcula a probabilidade de que o n-ésimo ponto de dados atrasado seja preditivo do atual. Se nenhum atraso máximo for dado, a função calcula um atraso máximo e retorna os valores de p para todos os atrasos até aquele. Podemos ver aqui que, para os 5 pontos de dados mais recentes, existe uma correlação significativa com a atual. Portanto, concluimos que ambos os dados são auto-correlacionados.


```python
_, prices_qstats, prices_qstat_pvalues = statsmodels.tsa.stattools.acf(y, qstat=True)
_, prices_qstats, prices_qstat_pvalues = statsmodels.tsa.stattools.acf(y-prediction, qstat=True)

print ('Valor-p para autocorrelação nos preços', prices_qstat_pvalues)
print ('Valor-p para autocorrelação nos residuais', prices_qstat_pvalues)
_, jb_pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(y-prediction)

print ('Valor-p para os residuais serem normalmente distribuidos', jb_pvalue)
```

    Valor-p para autocorrelação nos preços [1.42009969e-048 6.66502390e-086 3.23416224e-117 3.68954816e-143
     7.06464900e-163 4.23392897e-177 9.48186485e-188 6.41073849e-195
     1.04325249e-199 1.56726991e-202 1.08630027e-203 1.04767610e-203
     4.27875490e-203 3.08170794e-202 2.63004865e-201 1.73805504e-200
     6.30772922e-200 8.58030348e-200 2.48156240e-200 1.81662204e-201
     2.80699331e-203 6.35873113e-206 3.22455063e-209 4.87752904e-213
     7.42648699e-217 2.82197389e-220 1.55809639e-223 6.57813075e-227
     2.53668306e-230 3.09096416e-234 4.41080801e-239 1.52562269e-244
     5.89569196e-250 6.02406930e-256 2.28957060e-262 5.04866274e-268
     9.71500936e-273 1.17238783e-276 3.57781372e-280 4.67949014e-283]
    Valor-p para autocorrelação nos residuais [1.42009969e-048 6.66502390e-086 3.23416224e-117 3.68954816e-143
     7.06464900e-163 4.23392897e-177 9.48186485e-188 6.41073849e-195
     1.04325249e-199 1.56726991e-202 1.08630027e-203 1.04767610e-203
     4.27875490e-203 3.08170794e-202 2.63004865e-201 1.73805504e-200
     6.30772922e-200 8.58030348e-200 2.48156240e-200 1.81662204e-201
     2.80699331e-203 6.35873113e-206 3.22455063e-209 4.87752904e-213
     7.42648699e-217 2.82197389e-220 1.55809639e-223 6.57813075e-227
     2.53668306e-230 3.09096416e-234 4.41080801e-239 1.52562269e-244
     5.89569196e-250 6.02406930e-256 2.28957060e-262 5.04866274e-268
     9.71500936e-273 1.17238783e-276 3.57781372e-280 4.67949014e-283]
    Valor-p para os residuais serem normalmente distribuidos 0.006222791529906993


## Newey-West

Newey-West é um método de computar variância corrigido para autocorrelação. Uma computação de variação normal produzirá erros padrão imprecisos na presença de autocorrelação.

Podemos tentar alterar a equação de regressão para eliminar a autocorrelação. Uma solução mais simples é ajustar os erros padrão usando um método apropriado e usar os valores ajustados para verificar a significância. Abaixo, usamos o método Newey-West do `statsmodels` para calcular erros padrão ajustados para os coeficientes. Eles são mais altos do que os originalmente relatados pela regressão, que é o que esperamos para erros positivamente correlacionados.


```python
from math import sqrt

# Find the covariance matrix of the coefficients
cov_mat = stats.sandwich_covariance.cov_hac(model)

# Print the standard errors of each coefficient from the original model and from the adjustment
print ('Erro padrão comum:', model.bse[0], model.bse[1])
print ('Erro padrão ajustado:', sqrt(cov_mat[0,0]), sqrt(cov_mat[1,1]))
```

    Erro padrão comum: 0.6967678365574217 0.004861381292865307
    Erro padrão ajustado: 1.226176817259547 0.009667456045254657


# Multicollinearidade

Ao usar múltiplas variáveis independentes, é importante verificar a multicolinearidade; ou seja, uma relação linear aproximada entre as variáveis independentes, como, por exemplo
$$ X_2 \approx 5 X_1 - X_3 + 4.5 $$
Com a multicolinearidade, é difícil identificar o efeito independente de cada variável, pois podemos mudar em torno dos coeficientes de acordo com a relação linear sem alterar o modelo. Tal como acontece com variáveis verdadeiramente desnecessárias, isso geralmente não prejudicará a precisão do modelo, mas irá nublar nossa análise. Em particular, os coeficientes estimados terão grandes erros padrão. Os coeficientes também não representam mais o efeito parcial de cada variável, pois com a multicolinearidade não podemos mudar uma variável enquanto mantendo as outras constantes.

A alta correlação entre variáveis independentes é indicativa de multicolinearidade. No entanto, não é suficiente, uma vez que queremos detectar a correlação entre uma das variáveis e uma combinação linear das outras variáveis. Se tivermos estatísticas R-quadrados altas, mas baixas estatísticas-t nos coeficientes (o ajuste é bom, mas os coeficientes não são estimados com precisão) podemos suspeitar de multicolinearidade. Para resolver o problema, podemos retirar uma das variáveis independentes envolvidas na relação linear.

Por exemplo, usando dois índices de estoque, como nossas variáveis independentes, provavelmente levarão a multicolinearidade. Abaixo, podemos ver que a remoção de um deles melhora as estatísticas-t sem prejudicar o R-quadrado.

Outra coisa importante a determinar aqui é qual variável pode ser a casual. Se formularmos a hipótese de que o mercado influencie AAPL e AMZN, o mercado é a variável que devemos usar em nosso modelo preditivo.


```python
# Load pricing data for asset and two market indices
symbols = [
    'AAPL', # apple
    'AMZN', # amazon
    'SPY' # S&P500
]

stocks = {}
for symbol in symbols:
    stocks[symbol] = web.DataReader(symbol, 'robinhood').close_price.apply(float).values

# Run multiple linear regression
mlr = regression.linear_model.OLS(stocks['AMZN'], sm.add_constant(np.column_stack((stocks['SPY'], stocks['AAPL'])))).fit()

# Construct fit curve using dependent variables and estimated coefficients
mlr_prediction = mlr.params[0] + mlr.params[1]*stocks['SPY'] + mlr.params[2]*stocks['AAPL']

# Print regression statistics 
print('R-quadrado:', mlr.rsquared_adj)
print('estatística-t dos coeficientes:\n', mlr.tvalues)

# Plot asset and model
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.plot(stocks['AMZN'])
plt.plot(mlr_prediction)
plt.legend(['Ação', 'Modelo']);
plt.ylabel('Preço');
```

    R-quadrado: 0.7709851423145416
    estatística-t dos coeficientes:
     [-17.45538397  12.97007745   0.50135446]



![png](model_violation_files/model_violation_19_1.png)



```python
# Perform linear regression
slr = regression.linear_model.OLS(stocks['AMZN'], sm.add_constant(stocks['SPY'])).fit()
slr_prediction = slr.params[0] + slr.params[1] * stocks['SPY']

# Print fit statistics
print('R-quadrado:', slr.rsquared_adj)
print('estatística-t dos coeficientes:\n', slr.tvalues)

# Plot asset and model
plt.figure(figsize=(14,6))
plt.grid(linestyle='--')
plt.plot(stocks['AMZN'])
plt.plot(slr_prediction)
plt.legend(['Ação', 'Modelo']);
plt.ylabel('Preço');
```

    R-quadrado: 0.7716792740239338
    estatística-t dos coeficientes:
     [-18.03163229  28.96879978]



![png](model_violation_files/model_violation_20_1.png)


# Exemplo: quarteto de Anscombe

Anscombe construiu 4 conjuntos de dados que não só têm a mesma média e variância em cada variável, mas também o mesmo coeficiente de correlação, linha de regressão e valor de regressão R-quadrado. Abaixo, nós testamos esse resultado bem como planejamos os conjuntos de dados. Uma rápida olhada nos gráficos mostra que apenas o primeiro conjunto de dados satisfaz os pressupostos do modelo de regressão. Conseqüentemente, os altos valores de R-quadrados dos outros três não são significativos, o que concorda com a nossa intuição de que os outros três não são modelados de acordo com as linhas de melhor ajuste.


```python
from scipy.stats import pearsonr

# Construct Anscombe's arrays
x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
x3 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

# Perform linear regressions on the datasets
slr1 = regression.linear_model.OLS(y1, sm.add_constant(x1)).fit()
slr2 = regression.linear_model.OLS(y2, sm.add_constant(x2)).fit()
slr3 = regression.linear_model.OLS(y3, sm.add_constant(x3)).fit()
slr4 = regression.linear_model.OLS(y4, sm.add_constant(x4)).fit()

# Print regression coefficients, Pearson r, and R-squared for the 4 datasets
print('Cofficients:', slr1.params, slr2.params, slr3.params, slr4.params)
print('Pearson r:', pearsonr(x1, y1)[0], pearsonr(x2, y2)[0], pearsonr(x3, y3)[0], pearsonr(x4, y4)[0])
print('R-squared:', slr1.rsquared, slr2.rsquared, slr3.rsquared, slr4.rsquared)

# Plot the 4 datasets with their regression lines
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(14,8))
xs = np.arange(20)
ax1.plot(slr1.params[0] + slr1.params[1]*xs, 'r')
ax1.scatter(x1, y1)
ax1.set_xlabel('x1')
ax1.set_ylabel('y1')
ax2.plot(slr2.params[0] + slr2.params[1]*xs, 'r')
ax2.scatter(x2, y2)
ax2.set_xlabel('x2')
ax2.set_ylabel('y2')
ax3.plot(slr3.params[0] + slr3.params[1]*xs, 'r')
ax3.scatter(x3, y3)
ax3.set_xlabel('x3')
ax3.set_ylabel('y3')
ax4.plot(slr4.params[0] + slr4.params[1]*xs, 'r')
ax4.scatter(x4,y4)
ax4.set_xlabel('x4')
ax4.set_ylabel('y4');
```

    Cofficients: [3.00009091 0.50009091] [3.00090909 0.5       ] [3.00245455 0.49972727] [3.00172727 0.49990909]
    Pearson r: 0.81642051634484 0.8162365060002427 0.8162867394895981 0.816521436888503
    R-squared: 0.6665424595087751 0.6662420337274844 0.6663240410665592 0.6667072568984653



![png](model_violation_files/model_violation_22_1.png)


### References
* "Quantitative Investment Analysis", by DeFusco, McLeavey, Pinto, and Runkle
* https://www.quantopian.com/lectures/violations-of-regression-models
