import os
import pandas as pd
import numpy as np
import math
import datetime

from limpeza_dos_dados import Limpeza
from haversine import haversine

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging

# Configurando o logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
log = logger.info

os.chdir("../Scripts")
os.getcwd()


# Função para extrair dados
def extrair_dados() -> object:
    #  Objeto para limpeza dos dados
    limpeza = Limpeza()

    df_clean = (
        pd.read_parquet("../Dados/BikeData-Processed.parquet")
        .pipe(limpeza.valores_vazios)
        .pipe(limpeza.remove_outliers)
        .assign(

            distancia=lambda _df: _df.apply(
                lambda row: haversine(row['estacao_inicio_latitude'], row['estacao_inicio_longitude'],
                                      row['estacao_fim_latitude'], row['estacao_fim_longitude']), axis=1),
            # distancia=df.apply(lambda row: calculate_haversine(row), axis=1),
            dia_inicio=lambda _df: _df['inicio_viagem'].dt.day,
            mes_inicio=lambda _df: _df['inicio_viagem'].dt.month,
            ano_inicio=lambda _df: _df['inicio_viagem'].dt.year,
            dia_fim=lambda _df: _df['fim_viagem'].dt.day,
            mes_fim=lambda _df: _df['fim_viagem'].dt.month,
            ano_fim=lambda _df: _df['fim_viagem'].dt.year,
            hora_inicio=lambda _df: _df['inicio_viagem'].dt.hour,
            minuto_inicio=lambda _df: _df['inicio_viagem'].dt.minute,
            segundo_inicio=lambda _df: _df['inicio_viagem'].dt.second,
            hora_fim=lambda _df: _df['fim_viagem'].dt.hour,
            minuto_fim=lambda _df: _df['fim_viagem'].dt.minute,
            segundo_fim=lambda _df: _df['fim_viagem'].dt.second,
            dia_semana=lambda _df: _df['inicio_viagem'].dt.day_name(),
            idade=lambda _df: _df['ano_inicio'] - _df["ano_nascimento"],
        )
    )
    return df_clean


#=====================================#
#  Tarefas da Segunda semana Gustavo  #
#=====================================#


#=====================================#
#  Análise Exploratória de Dados:     #
#=====================================#
# Para visualizar as primeiras linhas do DataFrame e ter uma ideia dos dados:
# log(df.head)
# Para verificar as informações dos dados:
# log(f'informações dos dados{df.info}')
# log('# Para verificar estatísticas descritivas dos dados:')
# log(df.describe)
# Para verificar a quantidade de valores nulos em cada coluna:
# log(df.isnull().sum)
# Para verificar a quantidade de valores únicos em cada coluna:
# log(df.nunique)
# log('# Para verificar a correlação entre as variáveis:')
# log(df.corr)



# plt.imshow(df.corr(), cmap='coolwarm')
# plt.colorbar()
# plt.xticks(range(len(df.columns)), df.columns, rotation=90)
# plt.yticks(range(len(df.columns)), df.columns)
# plt.show()
#

#=====================================#
#  Análise Estatística                #
#=====================================#



#=====================================#
#  Visualização de Dados              #
#=====================================#


# Comece analisando os dados para compreender o uso e identificar erros.
# Realize todas as análises relevantes.
# Unificar o tratamento de dados
#================================================================
# Qual é a distribuição do uso de bicicletas (pickups)
# durante os dias da semana? Existem dias de pico?
#================================================================
# Qual é a distribuição do uso de bicicletas (pickups)
# durante os horários do dia? Existem horas de pico?
#================================================================
# Essa distribuição se mantém em todos os dias da semana?
# Há diferenças entre dias úteis e finais de semana?


def distribuicao_uso_bicicletas_dias_semana(df):
    """
    Esta função plota um gráfico de barras para mostrar a distribuição do uso de bicicletas ao longo da semana.
    """
    # Verifica se o DataFrame está vazio
    if df.empty:
        print("Erro: DataFrame vazio.")
        return None

    # Criando um dataframe para contar a distância percorrida por dia da semana
    pickups_por_dia_semana = df.groupby('dia_semana')['distancia'].count().reset_index()

    # Definindo a ordem dos dias da semana para aparecer no gráfico
    dias_da_semana = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pickups_por_dia_semana['dia_semana'] = pd.Categorical(pickups_por_dia_semana['dia_semana'],
                                                          categories=dias_da_semana, ordered=True)

    # Plotando o gráfico de barras
    fig = px.bar(pickups_por_dia_semana, x='dia_semana', y='distancia')

    return fig



def distribuicao_uso_bicicletas_horarios(df):
    pickups_por_horario = df.groupby('hora_inicio')['distancia'].count().reset_index()

    fig = px.histogram(pickups_por_horario, x="hora_inicio", y="distancia", nbins=24, width=700, height=400)
    fig.update_layout(
        title_text="Distribuição do uso das bicicletas por horário",
        xaxis_title_text="Horário de início da viagem",
        yaxis_title_text="Número de viagens",
        bargap=0.1,
    )
    fig.show()


def distribuicao_uso_bicicletas_dias_semana_horarios(df):
    # Cria um pivot table para contar o número de viagens por dia da semana e horário
    pivot_table = df.pivot_table(index='dia_semana', columns='hora_inicio', values='id', aggfunc='count')

    # Cria o gráfico de calor
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis'))

    # Define os eixos e o título
    fig.update_layout(
        title="Distribuição do uso de bicicletas por dia da semana e horário",
        xaxis_title="Horário",
        yaxis_title="Dia da semana",
        xaxis={'type': 'category'},
        yaxis={'type': 'category'}
    )

    # Mostra o gráfico
    fig.show()


def diferenca_dias_uteis_finais_semana(df):
    # Calcula a média de viagens por dia útil
    dias_uteis = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    media_dias_uteis = df[df['dia_semana'].isin(dias_uteis)]['id'].count() / len(dias_uteis)

    # Calcula a média de viagens por final de semana
    finais_semana = ['Saturday', 'Sunday']
    media_finais_semana = df[df['dia_semana'].isin(finais_semana)]['id'].count() / len(finais_semana)

    # Calcula a diferença entre as médias
    diferenca = media_dias_uteis - media_finais_semana

    return diferenca


# if __name__ == '__main__':
#     df = pd.DataFrame()  # Or define df as the result of another function call, if applicable
#     df = extrair_dados()
#     distribuicao_uso_bicicletas_dias_semana(df)
#     distribuicao_uso_bicicletas_horarios(df)
#     distribuicao_uso_bicicletas_dias_semana_horarios(df)
#     diferenca_dias_uteis_finais_semana(df)
