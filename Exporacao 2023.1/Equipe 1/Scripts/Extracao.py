import pandas as pd



def main():
    # ID do Google Drive
    FILE_ID = '1a8gzQg37aaDHBO6OBVMieOZmNU7jTOL1'
    
    #  Link do arquivo
    link = f'https://drive.google.com/uc?export=download&id={FILE_ID}'

    # Obtendo o ID do arquivo
    pd.read_parquet(link).to_parquet("../Dados/BikeData.parquet", index=False)
    
    
    
if __name__ == "__main__":
    main()