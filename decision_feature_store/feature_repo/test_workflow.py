import subprocess
import os
import pandas as pd
from feast import FeatureStore, FileSource
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

def run_demo():
    """
    Executa um fluxo de trabalho de demonstração para o repositório de features de decisão.
    """
    store = FeatureStore(repo_path=".")
    
    print("\n--- Executando 'feast apply' para registrar as definições ---")
    subprocess.run(["feast", "apply"], check=True)

    print("\n--- Buscando features históricas e salvando como um dataset ---")
    fetch_and_save_historical_features(store)

    print("\n--- Executando 'feast teardown' para limpar os recursos ---")
    # Comente a linha abaixo se quiser manter os dados para inspecionar na UI
    # subprocess.run(["feast", "teardown"], check=True)


def fetch_and_save_historical_features(store: FeatureStore):
    """
    Cria um dataframe de entidades, busca as features históricas associadas
    e salva o resultado como um SavedDataset.
    """
    # Carrega o dataframe de entidades a partir do arquivo de dados correto.
    entity_df = pd.read_parquet("data/dados_treinamento_transformados.parquet")

    print("Entity DataFrame (com a coluna 'event_timestamp'):")
    print(entity_df.head())

    # Busca as features usando o Feature Service definido no repositório.
    feature_service = store.get_feature_service("vaga_model_service")
    
    # O método get_historical_features retorna um RetrievalJob
    retrieval_job = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    )

    # Para consumir diretamente em um DataFrame:
    training_df = retrieval_job.to_df()

    print("DataFrame de Treinamento Final:")
    print(training_df.head())

    # Define onde o dataset de treinamento será salvo
    output_path = os.path.join(store.repo_path, "data", "training_dataset.parquet")
    print(f"\nSalvando o dataset de treinamento em {output_path}")
    
    # Cria um SavedDataset para que ele apareça na UI
    dataset = store.create_saved_dataset(
        from_=retrieval_job,
        name="vagas_training_dataset", # Este nome aparecerá na UI
        storage=SavedDatasetFileStorage(path=output_path),
        allow_overwrite=True, # Permite sobrescrever o arquivo se ele já existir
    )

    print("\nDataset salvo com sucesso! Execute 'feast ui' para visualizá-lo.")


if __name__ == "__main__":
    run_demo()
