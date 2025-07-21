from datetime import timedelta
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    ValueType,
)
from feast.types import Float64, Int64, String

# Define um projeto para o repositório de features
project = Project(name="decision_feature_store", description="Um projeto para features de decisão")

# Define uma entidade para a vaga.
# Corrigido para INT64, que corresponde ao tipo de dado de id_vaga.
vaga_entity = Entity(name="vaga", join_keys=["id_vaga"], value_type=ValueType.INT64)

decision_features_match_source = FileSource(
    name="decision_features_match_source",
    path="data/dados_treinamento_transformados.parquet",
    timestamp_field="event_timestamp",
)

feature_schema = [
    # Features TF-IDF
    *[Field(name=f"tfidf_{i}", dtype=Float64) for i in range(100)],

    # Features One-Hot Encoded
    Field(name="tipo_contratacao_clt_cotas", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_clt_full", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_clt_full_cooperado_estagio_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_clt_full_cooperado_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_clt_full_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_cooperado", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_cooperado_estagio_hunting_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_cooperado_estagio_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_cotas_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_full", dtype=Int64),
    Field(name="tipo_contratacao_clt_full_cooperado", dtype=Int64),
    Field(name="tipo_contratacao_clt_full_cooperado_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_clt_full_hunting", dtype=Int64),
    Field(name="tipo_contratacao_clt_full_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_cooperado", dtype=Int64),
    Field(name="tipo_contratacao_cooperado_hunting_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_cooperado_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_estagio", dtype=Int64),
    Field(name="tipo_contratacao_hunting", dtype=Int64),
    Field(name="tipo_contratacao_pj_autonomo", dtype=Int64),
    Field(name="tipo_contratacao_vazio", dtype=Int64),
    Field(name="nivel_profissional_analista", dtype=Int64),
    Field(name="nivel_profissional_especialista", dtype=Int64),
    Field(name="nivel_profissional_estagiario", dtype=Int64),
    Field(name="nivel_profissional_junior", dtype=Int64),
    Field(name="nivel_profissional_lider", dtype=Int64),
    Field(name="nivel_profissional_pleno", dtype=Int64),
    Field(name="nivel_profissional_senior", dtype=Int64),
    Field(name="nivel_profissional_vazio", dtype=Int64),
    Field(name="nivel_academico_vazio", dtype=Int64),
    Field(name="nivel_ingles_avancado", dtype=Int64),
    Field(name="nivel_ingles_basico", dtype=Int64),
    Field(name="nivel_ingles_fluente", dtype=Int64),
    Field(name="nivel_ingles_intermediario", dtype=Int64),
    Field(name="nivel_ingles_nenhum", dtype=Int64),
    Field(name="nivel_ingles_vazio", dtype=Int64),
    Field(name="nivel_espanhol_avancado", dtype=Int64),
    Field(name="nivel_espanhol_basico", dtype=Int64),
    Field(name="nivel_espanhol_fluente", dtype=Int64),
    Field(name="nivel_espanhol_intermediario", dtype=Int64),
    Field(name="nivel_espanhol_nenhum", dtype=Int64),
    Field(name="nivel_espanhol_vazio", dtype=Int64),
    Field(name="ingles_vaga_avancado", dtype=Int64),
    Field(name="ingles_vaga_basico", dtype=Int64),
    Field(name="ingles_vaga_fluente", dtype=Int64),
    Field(name="ingles_vaga_intermediario", dtype=Int64),
    Field(name="ingles_vaga_nenhum", dtype=Int64),
    Field(name="ingles_vaga_tecnico", dtype=Int64),
    Field(name="ingles_vaga_vazio", dtype=Int64),
    Field(name="espanhol_vaga_avancado", dtype=Int64),
    Field(name="espanhol_vaga_basico", dtype=Int64),
    Field(name="espanhol_vaga_fluente", dtype=Int64),
    Field(name="espanhol_vaga_intermediario", dtype=Int64),
    Field(name="espanhol_vaga_nenhum", dtype=Int64),
    Field(name="espanhol_vaga_tecnico", dtype=Int64),
    Field(name="espanhol_vaga_vazio", dtype=Int64),
    Field(name="nivel_academico_vaga_vazio", dtype=Int64),

    # Features de Engenharia
    Field(name="match_ingles", dtype=Int64),
    Field(name="match_nivel_academico", dtype=Int64),
    Field(name="match_area_atuacao", dtype=Int64),
    Field(name="match_localidade", dtype=Int64),
    Field(name="match_pcd", dtype=Int64),
    Field(name="qtd_keywords_cv", dtype=Int64),
    Field(name="match_cv_atividade", dtype=Float64),
]


# Define uma Feature View para servir dados do parquet.
decision_features_fv = FeatureView(
    name="decision_features_match",
    entities=[vaga_entity],
    ttl=timedelta(days=30),
    schema=feature_schema,
    online=False,
    source=decision_features_match_source,
    tags={"team": "recruiting"},
)

# Agrupa as features em um serviço para o modelo consumir.
vaga_service = FeatureService(
    name="vaga_model_service",
    features=[
        decision_features_fv,
    ],
)

# ...existing code...

# Novo FileSource para o parquet melhorado
decision_features_clustering_source = FileSource(
    name="decision_features_clustering_source",
    path="data/processed_data_3_clusters_melhorado.parquet",
    timestamp_field="event_timestamp",
)

# Novas features do parquet melhorado
processed_feature_schema = [
    Field(name="match_score_semantico", dtype=Float64),
    Field(name="match_score_normalizado", dtype=Float64),
    Field(name="match_score_combinado", dtype=Float64),
    Field(name="skills_comuns", dtype=Float64),
    Field(name="cv_skills_count", dtype=Float64),
    Field(name="diversidade_skills", dtype=Float64),
    Field(name="taxa_sucesso_recrutador", dtype=Float64),
    Field(name="experiencia_recrutador", dtype=Float64),
    Field(name="taxa_sucesso_modalidade", dtype=Float64),
    Field(name="concorrencia_vaga", dtype=Float64),
    Field(name="log_concorrencia", dtype=Float64),
    Field(name="cv_quality_score", dtype=Float64),
    Field(name="completude_perfil", dtype=Float64),
    Field(name="mes_candidatura", dtype=Float64),
    Field(name="trimestre", dtype=Float64),
    Field(name="dia_semana", dtype=Float64),
    Field(name="is_inicio_ano", dtype=Float64),
    Field(name="is_meio_ano", dtype=Float64),
    Field(name="is_fim_ano", dtype=Float64),
    Field(name="tempo_perfil_candidatura", dtype=Int64),
    Field(name="tempo_vaga_candidatura", dtype=Float64),
    Field(name="match_x_experiencia_recrutador", dtype=Float64),
    Field(name="match_x_taxa_sucesso_recrutador", dtype=Float64),
    Field(name="skills_x_taxa_sucesso_modalidade", dtype=Float64),
    Field(name="concorrencia_x_match", dtype=Float64),
    Field(name="match_per_concorrencia", dtype=Float64),
    Field(name="skills_per_experiencia", dtype=Float64),
    Field(name="modalidade_encoded", dtype=Float64),
    Field(name="recrutador_encoded", dtype=Float64),
    Field(name="sexo_encoded", dtype=Float64),
    Field(name="estado_civil_encoded", dtype=Float64),
    Field(name="nivel_academico_encoded", dtype=Float64),
    Field(name="nivel_ingles_encoded", dtype=Float64),
    Field(name="nivel_profissional_encoded", dtype=Float64),
    Field(name="tipo_contratacao_encoded", dtype=Float64),
    Field(name="cliente_encoded", dtype=Float64),
    Field(name="requisitante_encoded", dtype=Float64),
    Field(name="pais_encoded", dtype=Float64),
    Field(name="estado_vaga_encoded", dtype=Float64),
    Field(name="cidade_vaga_encoded", dtype=Float64),
    Field(name="cv_tem_experiencia", dtype=Float64),
    Field(name="cv_tem_projeto", dtype=Float64),
    Field(name="cv_tem_desenvolvimento", dtype=Float64),
    Field(name="cv_tem_gestao", dtype=Float64),
    Field(name="cv_tem_lideranca", dtype=Float64),
    Field(name="cv_tem_equipe", dtype=Float64),
    Field(name="cv_tem_resultado", dtype=Float64),
    Field(name="cluster_target", dtype=Int64),
]

# Nova FeatureView para as novas features
processed_features_fv = FeatureView(
    name="decision_features_clustering",
    entities=[vaga_entity],
    ttl=timedelta(days=30),
    schema=processed_feature_schema,
    online=False,
    source=decision_features_clustering_source,
    tags={"team": "recruiting"},
)

# Atualize o FeatureService para incluir as duas FeatureViews
vaga_service = FeatureService(
    name="vaga_model_service",
    features=[
        decision_features_fv,
        processed_features_fv,
    ],
)

