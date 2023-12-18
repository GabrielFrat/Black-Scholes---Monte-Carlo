import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sklearn as sl
import tables as tb
import math
from scipy.stats import norm
import requests
import yfinance as yf
from datetime import datetime, timedelta


# =========================================================================
# importar a taxa SELIC
# =========================================================================
# busca o histórico diário da selic
r = requests.get('https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json').json()
dfSelic = pd.DataFrame(r).set_index('data')
dfSelic['valor'] = dfSelic['valor'].astype(float)

# anualiza as taxas, considerando 252 dias úteis no ano
dfSelic['valor'] = dfSelic['valor'].apply(lambda x: 1+x/100)**252-1


# obtem o valor atual da taxa de juros
r = dfSelic['valor'].iloc[-1]
print("Valor da taxa de juros: ", r)


# =========================================================================
# Buscar as cotações
# =========================================================================
ticker = "KLBN11.SA"
# busca o ticker no yahoo finance
dias = 365
start = (datetime.today()-timedelta(dias)).strftime('%Y-%m-%d')
dfCotacoes = yf.download(ticker, start)
print(dfCotacoes)

K = 22.5
S0 = dfCotacoes['Adj Close'][-1]
print("Valor da acao", S0)

# Gera a coluna de retorno os retornos logaritmicos
dfCotacoes['Returns'] = np.log(dfCotacoes['Adj Close']/dfCotacoes['Adj Close'].shift(1))
sigma = dfCotacoes['Returns'].std() * 252 ** 0.5
print("Volatilidade do ativo: ", sigma)

vencimento = '2024-01-19'
T = (datetime.strptime(vencimento, '%Y-%m-%d') - datetime.now()).days/365



# Algoritimo de precificação BS
d1 = (np.log(S0/K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
d2 = (np.log(S0/K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
call = (S0 * norm.cdf(d1) - (K * np.exp(-r * T) * norm.cdf(d2)))
print('O valor da opção é %5.3f.' % call)

# Algoritimo de precificação por monte carlo
I = 100000
z = np.random.standard_normal(I)
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
hT = np.maximum(ST - K, 0)
C0 = math.exp(-r * T) * np.mean(hT)

print('O valor da opção é %5.3f.' % C0)

