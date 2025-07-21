import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from feature_engineering.data_preprocessing import pre_processar


def criar_coluna_contratado_refinada(df):
    """
    Refina a coluna 'contratado' com base na situação do candidato e separa o dataset
    entre dados de treinamento (com rótulo definido) e dados em andamento (sem rótulo).

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna 'situacao_candidado'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_treinamento: Subconjunto com a coluna 'contratado' preenchida (0 ou 1).
            - df_em_andamento: Subconjunto com candidatos sem definição final.
    """
    contratado_status = [
        'contratado pela decision',
        'contratado como hunting',
        'proposta aceita'
    ]
    nao_contratado_status = [
        'nao aprovado pelo cliente',
        'desistiu',
        'nao aprovado pelo rh',
        'nao aprovado pelo requisitante',
        'sem interesse nesta vaga',
        'desistiu da contratacao',
        'recusado'
    ]

    df['contratado'] = np.nan
    df.loc[df['situacao_candidado'].isin(contratado_status), 'contratado'] = 1
    df.loc[df['situacao_candidado'].isin(nao_contratado_status), 'contratado'] = 0

    df_treinamento = df.dropna(subset=['contratado']).copy()
    df_treinamento['contratado'] = df_treinamento['contratado'].astype(int)

    df_em_andamento = df[df['contratado'].isna()].copy()
    if 'contratado' in df_em_andamento.columns:
        df_em_andamento.drop(columns=['contratado'], inplace=True)

    return df_treinamento, df_em_andamento


def carregar_dados(path):
    """
    Carrega dados a partir de um arquivo no formato Parquet.
    """
    return pd.read_parquet(path)


def extrair_e_transformar_features(df_input, tfidf_model=None, ohe_models=None, original_feature_columns=None, is_training=True):
    """
    Extrai e transforma um conjunto completo de features a partir de dados textuais e estruturados,
    aplicando vetorização TF-IDF e One-Hot Encoding.

    Args:
        df_input (pd.DataFrame): DataFrame contendo as colunas de texto e estruturadas.
        tfidf_model (TfidfVectorizer, opcional): Modelo TF-IDF previamente ajustado.
        ohe_models (dict, opcional): Dicionário de objetos OneHotEncoder previamente ajustados.
        original_feature_columns (list, opcional): Lista de nomes das colunas de features esperadas.
        is_training (bool): Indica se a função está sendo chamada para treino (True) ou predição (False).

    Returns:
        Tuple[pd.DataFrame, TfidfVectorizer, dict, list]:
            - DataFrame com as features combinadas.
            - O objeto TfidfVectorizer.
            - O dicionário de objetos OneHotEncoder.
            - A lista de nomes das colunas finais de features.
    """
    df = df_input.copy()
    df.columns = df.columns.astype(str)

    df = pre_processar(df)

    texto_completo = (
            df['cv'].fillna('') + ' ' +
            df['objetivo_profissional'].fillna('') + ' ' +
            df['titulo_profissional'].fillna('') + ' ' +
            df['principais_atividades_vaga'].fillna('')
    )

    if is_training:
        tfidf = TfidfVectorizer(max_features=100)
        X_texto = tfidf.fit_transform(texto_completo)
    else:
        if tfidf_model is None:
            raise ValueError("tfidf_model deve ser fornecido para predição/teste.")
        tfidf = tfidf_model
        X_texto = tfidf.transform(texto_completo)

    X_texto_df = pd.DataFrame(X_texto.toarray(), columns=[f'tfidf_{i}' for i in range(X_texto.shape[1])])

    cols_to_encode = [
        "tipo_contratacao", "nivel_profissional", "nivel_academico",
        "nivel_ingles", "nivel_espanhol", "ingles_vaga", "espanhol_vaga",
        "nivel_academico_vaga"
    ]

    ohe_fitted_models = {}
    df_encoded_features = pd.DataFrame(index=df.index)

    for col in cols_to_encode:
        if is_training:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = ohe.fit_transform(df[[col]])
            ohe_fitted_models[col] = ohe
        else:
            if ohe_models is None or col not in ohe_models:
                raise ValueError(f"OneHotEncoder para a coluna '{col}' deve ser fornecido para predição/teste.")
            ohe = ohe_models[col]
            encoded_data = ohe.transform(df[[col]])

        new_cols_names = ohe.get_feature_names_out([col])
        temp_df = pd.DataFrame(encoded_data, columns=new_cols_names, index=df.index)
        df_encoded_features = pd.concat([df_encoded_features, temp_df], axis=1)

    X_numeric_binary = df.filter(
        regex=r'^(match_ingles|match_nivel_academico|match_area_atuacao|match_localidade|match_pcd|qtd_keywords_cv|match_cv_atividade$)'
    ).reset_index(drop=True)

    X_final = pd.concat([X_texto_df, df_encoded_features.reset_index(drop=True), X_numeric_binary], axis=1)
    X_final.columns = X_final.columns.astype(str)

    if is_training:
        final_feature_columns = X_final.columns.tolist()
    else:
        if original_feature_columns is None:
            raise ValueError("original_feature_columns deve ser fornecido para predição/teste.")
        X_final = X_final.reindex(columns=original_feature_columns, fill_value=0)
        final_feature_columns = original_feature_columns

    return X_final, tfidf, ohe_fitted_models, final_feature_columns


def run_feature_engineering_pipeline(data_path: str, artifacts_path: str, output_path: str):
    """
    Executa o pipeline de engenharia de features, salvando os dados transformados e os artefatos.

    Args:
        data_path (str): Caminho para o arquivo de dados de entrada.
        artifacts_path (str): Diretório para salvar os artefatos (transformadores).
        output_path (str): Diretório para salvar os dados de saída transformados.
    """
    print("Iniciando pipeline de engenharia de features...")
    df = carregar_dados(data_path)
    df.columns = df.columns.astype(str)

    # Garante que id_vaga existe
    if 'id_vaga' not in df.columns:
        raise ValueError("A coluna 'id_vaga' é necessária e não foi encontrada no dataframe.")

    print("1. Separando dados de treinamento e em andamento...")
    df_treinamento, df_em_andamento = criar_coluna_contratado_refinada(df)

    print("2. Extraindo e transformando features do conjunto de treinamento...")
    X_treino, tfidf_model, ohe_models, feature_columns = extrair_e_transformar_features(
        df_treinamento, is_training=True
    )
    y_treino = df_treinamento['contratado']

    os.makedirs(artifacts_path, exist_ok=True)
    joblib.dump(ohe_models, os.path.join(artifacts_path, "one_hot_encoders.pkl"))
    joblib.dump(feature_columns, os.path.join(artifacts_path, "feature_columns.pkl"))
    print(f"Artefatos de transformação salvos em: {artifacts_path}")

    os.makedirs(output_path, exist_ok=True)
    
    # Resgata 'id_vaga' e o alvo do dataframe de treinamento original
    ids_e_target = df_treinamento[['id_vaga', 'contratado']].reset_index(drop=True)
    df_treino_final = pd.concat([ids_e_target, X_treino], axis=1)

    # Cria a coluna 'event_timestamp' com o valor de 'data_candidatura' em formato datetime
    df_treino_final['event_timestamp'] = pd.to_datetime(df['data_candidatura'], format='%d-%m-%Y')
    output_file_treino = os.path.join(output_path, "dados_treinamento_transformados.parquet")
    df_treino_final.to_parquet(output_file_treino, index=False)
    print(f"Dados de treinamento transformados salvos em: {output_file_treino}")

    if not df_em_andamento.empty:
        print("3. Transformando features do conjunto 'em andamento'...")
        X_em_andamento, _, _, _ = extrair_e_transformar_features(
            df_em_andamento,
            tfidf_model=tfidf_model,
            ohe_models=ohe_models,
            original_feature_columns=feature_columns,
            is_training=False
        )
        
        # Resgata 'id_vaga' do dataframe "em andamento" original
        ids_em_andamento = df_em_andamento[['id_vaga']].reset_index(drop=True)
        df_em_andamento_final = pd.concat([ids_em_andamento, X_em_andamento], axis=1)

        output_file_em_andamento = os.path.join(output_path, "dados_em_andamento_transformados.parquet")
        df_em_andamento_final.to_parquet(output_file_em_andamento, index=False)
        print(f"Dados 'em andamento' transformados salvos em: {output_file_em_andamento}")

    print("\nPipeline de engenharia de features concluído com sucesso!")


if __name__ == "__main__":
    BASE_DIR = '/home/d3v0tchk4/Documents/repos/decision_feature_store/feature_repo/'
    DATA_PATH = os.path.join(BASE_DIR, 'data/', 'decision_features.parquet')
    ARTIFACTS_PATH = os.path.join(BASE_DIR, 'artifacts/')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'data/')

    run_feature_engineering_pipeline(
        data_path=DATA_PATH,
        artifacts_path=ARTIFACTS_PATH,
        output_path=OUTPUT_PATH
    )
