import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

import logging

# Configurando o logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
log = logger.info


# =================================================================
# Limpeza dos Dados
# =================================================================


class Limpeza:
    def __init__(self):

        '''
        tripduration                0
        starttime                   0
        stoptime                    0
        start station id           73
        start station name         73
        start station latitude      0
        start station longitude     0
        end station id             73
        end station name           73
        end station latitude        0
        end station longitude       0
        bikeid                      0
        usertype                    0
        birth year                  0
        gender                      0
        dtype: int64

        '''
        pass

    @staticmethod
    def valores_vazios(dataframe):
        """
        Identifica e trata valores vazios em um dataframe.

        Argumentos:
        df -- um dataframe

        Retorno:
        Um dataframe sem valores vazios.
        """
        # Identificando valores vazios
        log('Identificando valores vazios...')
        log(dataframe.isnull().sum())

        # Tratando valores vazios
        log('Tratando valores vazios...')
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_imputed = pd.DataFrame(imp_mean.fit_transform(dataframe), columns=dataframe.columns)

        log('Valores vazios tratados.')
        return df_imputed

    @staticmethod
    def outliers(dataframe, cols=None):
        """
        Identifica e trata outliers em um dataframe.

        Argumentos:
        df -- um dataframe
        cols -- uma lista com as colunas a serem analisadas. Se não for informado, serão analisadas todas as colunas.

        Retorno:
        Um dataframe sem outliers.
        """
        log('Identificando outliers...')
        if cols is None:
            cols = dataframe.columns

        # Usando o método de Tukey para identificar outliers
        for col in cols:
            q1 = dataframe[col].quantile(0.25)
            q3 = dataframe[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outliers = dataframe[(dataframe[col] < low) | (dataframe[col] > high)]
            if len(outliers) > 0:
                logger.info(f'{col}: {len(outliers)} outliers encontrados.')
                logger.info(outliers)

        # Removendo outliers
        log('Removendo outliers...')
        for col in cols:
            q1 = dataframe[col].quantile(0.25)
            q3 = dataframe[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            dataframe = dataframe[(dataframe[col] >= low) & (dataframe[col] <= high)]

        log('Outliers removidos.')
        return dataframe

    def normalizacao_zscore(self, col):
        # Normaliza os dados de uma coluna utilizando a técnica Z-Score
        scaler = StandardScaler()
        self.df[col] = scaler.fit_transform(self.df[col].values.reshape(-1, 1))
        log('Coluna {} normalizada utilizando Z-Score'.format(col))


if __name__ == '__main__':
    # import dataset
    df = pd.read_parquet("../Dados/BikeData-Raw.parquet")
    log(df.info)

    # Identificando valores vazios
    nulos = df.isna().sum()
    log('Quantidade de nulos: {}'.format(len(nulos)))
    log('Valores nulos:\n {}'.format(nulos))

    # Preencher com valores fixos:
    # Media



    # classe para limpeza dos dados
    limpeza = Limpeza()
    # limpeza.valores_vazios(df)
