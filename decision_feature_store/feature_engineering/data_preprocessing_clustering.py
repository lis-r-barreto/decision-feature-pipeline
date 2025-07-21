#!/usr/bin/env python3
"""
PROCESSAMENTO MELHORADO FINAL - 3 CLUSTERS
Implementa todas as melhorias criticas para 3 clusters:
- Matching semantico com TF-IDF
- Features temporais avancadas  
- Features de performance historica
- Features de interacao complexas
- Features de qualidade do candidato

CLUSTERS:
Cluster 0: Candidatos de Sucesso (Finalizaram o Funil com Exito)
Cluster 1: Candidatos que Sairam do Processo (Desistiram ou Foram Reprovados)
Cluster 2: Candidatos em Processo (Ativos ou Pendentes)
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ProcessadorDados3ClustersMelhorado:
    def __init__(self):
        self.data_dir = './decision_feature_store/feature_repo/data'
        self.artifacts_dir = './decision_feature_store/feature_repo/artifacts'
        self.reports_dir = './decision_feature_store/feature_repo/reports'
        
        # Criar diretorios se nao existirem
        for directory in [self.artifacts_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Mapeamento dos 3 clusters
        self.clusters_mapping = {
            # CLUSTER 0: CANDIDATOS DE SUCESSO (Finalizaram o Funil com Exito)
            'Contratado pela Decision': 0,
            'Contratado como Hunting': 0,
            'Aprovado': 0,
            
            # CLUSTER 1: CANDIDATOS QUE SAIRAM DO PROCESSO (Desistiram ou Foram Reprovados)
            'Nao Aprovado pelo Cliente': 1,
            'Nao Aprovado pelo RH': 1,
            'Nao Aprovado pelo Requisitante': 1,
            'Recusado': 1,
            'Desistiu': 1,
            'Sem interesse nesta vaga': 1,
            'Desistiu da Contratacao': 1,
            
            # CLUSTER 2: CANDIDATOS EM PROCESSO (Ativos ou Pendentes)
            'Prospect': 2,
            'Inscrito': 2,
            'Em avaliacao pelo RH': 2,
            'Encaminhado ao Requisitante': 2,
            'Entrevista Tecnica': 2,
            'Entrevista com Cliente': 2,
            'Documentacao PJ': 2,
            'Documentacao CLT': 2,
            'Documentacao Cooperado': 2,
            'Encaminhar Proposta': 2,
            'Proposta Aceita': 2
        }
        
        # Skills tecnicas expandidas
        self.skills_tecnicas = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'machine learning', 'data science', 'tensorflow', 'pytorch',
            'scrum', 'agile', 'devops', 'ci/cd', 'microservices', 'api',
            'html', 'css', 'bootstrap', 'sass', 'webpack', 'typescript',
            'spring', 'django', 'flask', 'express', 'laravel', 'rails',
            'android', 'ios', 'flutter', 'react native', 'xamarin',
            'tableau', 'power bi', 'excel', 'r', 'scala', 'spark'
        ]
        
        # Inicializar componentes
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        
    def flatten_json(self, data, prefix='', separator='_'):
        """Achata estruturas JSON aninhadas"""
        flattened = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    flattened.update(self.flatten_json(value, new_key, separator))
                else:
                    flattened[new_key] = value
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{prefix}{separator}{i}" if prefix else str(i)
                if isinstance(item, (dict, list)):
                    flattened.update(self.flatten_json(item, new_key, separator))
                else:
                    flattened[new_key] = item
        else:
            flattened[prefix] = data
            
        return flattened
    
    def extrair_dados_prospects(self, prospects_data):
        """Extrai dados dos prospects com flattening"""
        dados = []
        
        for codigo_vaga, vaga_info in prospects_data.items():
            titulo_vaga = vaga_info.get('titulo', '')
            modalidade = vaga_info.get('modalidade', '')
            
            for prospect in vaga_info.get('prospects', []):
                # Fazer flattening do prospect
                prospect_flat = self.flatten_json(prospect, prefix='prospect')
                
                registro = {
                    'codigo_vaga': codigo_vaga,
                    'codigo_candidato': prospect.get('codigo', ''),
                    'situacao_candidado': prospect.get('situacao_candidado', ''),
                    'data_candidatura': prospect.get('data_candidatura', ''),
                    'recrutador': prospect.get('recrutador', ''),
                    'titulo_vaga_prospect': titulo_vaga,
                    'modalidade': modalidade
                }
                
                # Adicionar campos flattened
                registro.update(prospect_flat)
                dados.append(registro)
        
        return pd.DataFrame(dados)
    
    def extrair_dados_applicants(self, applicants_data):
        """Extrai dados dos applicants com flattening completo"""
        dados = []
        
        for codigo_candidato, candidato_info in applicants_data.items():
            # Fazer flattening completo do candidato
            candidato_flat = self.flatten_json(candidato_info, prefix='applicant')
            
            # Extrair secoes estruturadas
            infos_pessoais = candidato_info.get('informacoes_pessoais', {})
            infos_profissionais = candidato_info.get('informacoes_profissionais', {})
            formacao_idiomas = candidato_info.get('formacao_idiomas', {})
            
            registro = {
                'codigo_candidato': codigo_candidato,
                
                # Informacoes pessoais
                'nome': infos_pessoais.get('nome', ''),
                'data_nascimento': infos_pessoais.get('data_nascimento', ''),
                'sexo': infos_pessoais.get('sexo', ''),
                'estado_civil': infos_pessoais.get('estado_civil', ''),
                'endereco': infos_pessoais.get('endereco', ''),
                'cpf': infos_pessoais.get('cpf', ''),
                'email_secundario': infos_pessoais.get('email_secundario', ''),
                'telefone_celular': infos_pessoais.get('telefone_celular', ''),
                'url_linkedin': infos_pessoais.get('url_linkedin', ''),
                
                # Informacoes profissionais
                'titulo_profissional': infos_profissionais.get('titulo_profissional', ''),
                'area_atuacao': infos_profissionais.get('area_atuacao', ''),
                'conhecimentos_tecnicos': infos_profissionais.get('conhecimentos_tecnicos', ''),
                'certificacoes': infos_profissionais.get('certificacoes', ''),
                'remuneracao': infos_profissionais.get('remuneracao', ''),
                'nivel_profissional': infos_profissionais.get('nivel_profissional', ''),
                'objetivo_profissional': infos_profissionais.get('objetivo_profissional', ''),
                
                # Formacao e idiomas
                'nivel_academico': formacao_idiomas.get('nivel_academico', ''),
                'nivel_ingles': formacao_idiomas.get('nivel_ingles', ''),
                'nivel_espanhol': formacao_idiomas.get('nivel_espanhol', ''),
                'outro_idioma': formacao_idiomas.get('outro_idioma', ''),
                
                # CV - CAMPO PRINCIPAL!
                'cv_texto': candidato_info.get('cv_pt', ''),
                'cv_en': candidato_info.get('cv_en', ''),
                'data_criacao': candidato_info.get('data_criacao', '')
            }
            
            # Adicionar campos flattened
            registro.update(candidato_flat)
            dados.append(registro)
        
        return pd.DataFrame(dados)
    
    def extrair_dados_vagas_completo(self, vagas_data):
        """Extrai dados das vagas com flattening completo"""
        dados = []
        
        for codigo_vaga, vaga_info in vagas_data.items():
            if not isinstance(vaga_info, dict):
                continue
                
            # Fazer flattening completo da vaga
            vaga_flat = self.flatten_json(vaga_info, prefix='vaga')
            
            # Extrair informacoes estruturadas
            info_basicas = vaga_info.get('informacoes_basicas', {})
            perfil = vaga_info.get('perfil_vaga', {})
            beneficios = vaga_info.get('beneficios', {})
            
            registro = {
                'codigo_vaga': codigo_vaga,
                
                # Informacoes basicas
                'data_requicisao': info_basicas.get('data_requicisao', ''),
                'limite_esperado_para_contratacao': info_basicas.get('limite_esperado_para_contratacao', ''),
                'titulo_vaga': info_basicas.get('titulo_vaga', ''),
                'cliente': info_basicas.get('cliente', ''),
                'requisitante': info_basicas.get('requisitante', ''),
                'tipo_contratacao': info_basicas.get('tipo_contratacao', ''),
                'objetivo_vaga': info_basicas.get('objetivo', ''),
                'prioridade_vaga': info_basicas.get('prioridade', ''),
                'pais': info_basicas.get('pais', ''),
                'estado_vaga': info_basicas.get('estado', ''),
                'cidade_vaga': info_basicas.get('cidade', ''),
                
                # Perfil da vaga
                'principais_atividades': perfil.get('principais_atividades', ''),
                'competencia_tecnicas_e_comportamentais': perfil.get('competencia_tecnicas_e_comportamentais', ''),
                'nivel_academico_vaga': perfil.get('nivel_academico', ''),
                'nivel_ingles_vaga': perfil.get('nivel_ingles', ''),
                'nivel_profissional_vaga': perfil.get('nivel_profissional', ''),
                'experiencia_minima': perfil.get('experiencia_minima', ''),
                'experiencia_maxima': perfil.get('experiencia_maxima', ''),
                
                # Beneficios
                'salario_minimo': beneficios.get('salario_minimo', ''),
                'salario_maximo': beneficios.get('salario_maximo', ''),
                'beneficios_oferecidos': beneficios.get('beneficios', '')
            }
            
            # Adicionar campos flattened
            registro.update(vaga_flat)
            dados.append(registro)
        
        return pd.DataFrame(dados)
    
    def load_and_prepare_data(self):
        """Carrega e prepara dados com merge"""
        print("=== CARREGAMENTO E PREPARACAO DOS DADOS ===")
        
        # Carregar dados
        with open(os.path.join(self.data_dir, 'prospects.json'), 'r', encoding='utf-8') as f:
            prospects_data = json.load(f)
        
        with open(os.path.join(self.data_dir, 'vagas.json'), 'r', encoding='utf-8') as f:
            vagas_data = json.load(f)
        
        with open(os.path.join(self.data_dir, 'applicants.json'), 'r', encoding='utf-8') as f:
            applicants_data = json.load(f)
        
        print(f"Prospects: {len(prospects_data)} vagas com candidatos")
        print(f"Vagas: {len(vagas_data)} registros")
        print(f"Applicants: {len(applicants_data)} candidatos")
        
        print("\nExtraindo e unindo dados com flattening...")
        
        # Extrair dados
        df_prospects = self.extrair_dados_prospects(prospects_data)
        df_applicants = self.extrair_dados_applicants(applicants_data)
        df_vagas = self.extrair_dados_vagas_completo(vagas_data)
        
        print(f"Prospects extraidos: {len(df_prospects)} candidaturas")
        print(f"Applicants extraidos: {len(df_applicants)} candidatos")
        print(f"Vagas extraidas: {len(df_vagas)} vagas")
        
        print("\nRealizando merge dos dados...")
        
        # Merge dos dados
        df_merged = df_prospects.merge(df_applicants, on='codigo_candidato', how='left')
        print(f"Apos merge prospects + applicants: {len(df_merged)} registros")
        
        df_final = df_merged.merge(df_vagas, on='codigo_vaga', how='left')
        print(f"Dataset final apos merge: {len(df_final)} registros x {len(df_final.columns)} colunas")
        
        return df_final
    
    def criar_clusters_target(self, df):
        """Cria variavel target com 3 clusters especificos"""
        print("\n=== CRIANDO CLUSTERS TARGET (3 CLUSTERS) ===")
        
        # Mapear situacoes para clusters
        df['cluster_target'] = df['situacao_candidado'].map(self.clusters_mapping)
        
        # Verificar mapeamento
        situacoes_nao_mapeadas = df[df['cluster_target'].isna()]['situacao_candidado'].unique()
        if len(situacoes_nao_mapeadas) > 0:
            print(f"Situacoes nao mapeadas: {situacoes_nao_mapeadas}")
            # Mapear para cluster 2 (em processo) por padrao
            df['cluster_target'] = df['cluster_target'].fillna(2)
        
        # Mostrar distribuicao
        print("\nDISTRIBUICAO DOS 3 CLUSTERS FINAIS:")
        distribuicao = df['cluster_target'].value_counts().sort_index()
        total = len(df)
        
        clusters_nomes = {
            0: "CANDIDATOS DE SUCESSO",
            1: "CANDIDATOS QUE SAIRAM DO PROCESSO", 
            2: "CANDIDATOS EM PROCESSO"
        }
        
        for cluster_id, count in distribuicao.items():
            pct = (count / total) * 100
            nome = clusters_nomes.get(cluster_id, f"CLUSTER {cluster_id}")
            print(f"   Cluster {cluster_id} - {nome}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def limpeza_basica_melhorada(self, df):
        """Limpeza basica dos dados sem remover registros importantes"""
        print("\n=== LIMPEZA BASICA MELHORADA ===")
        
        # Preencher CVs vazios em vez de remover
        df['cv_texto'] = df['cv_texto'].fillna('CV nao disponivel')
        df.loc[df['cv_texto'].str.len() <= 10, 'cv_texto'] = 'CV nao disponivel'
        
        print(f"CVs vazios preenchidos com texto padrao")
        
        # Remover duplicatas
        duplicatas_antes = len(df)
        df = df.drop_duplicates(subset=['codigo_candidato', 'codigo_vaga'])
        duplicatas_depois = len(df)
        duplicatas_removidas = duplicatas_antes - duplicatas_depois
        
        if duplicatas_removidas > 0:
            print(f"{duplicatas_removidas} duplicatas removidas")
        else:
            print("Nenhuma duplicata encontrada")
        
        print(f"Registros mantidos: {len(df)}")
        
        return df
    
    def criar_features_matching_semantico(self, df):
        """Cria features de matching semantico avancado com TF-IDF"""
        print("\nCRIANDO MATCHING SEMANTICO AVANCADO...")
        
        # Preparar textos
        df['cv_texto'] = df['cv_texto'].fillna('').astype(str)
        df['competencia_tecnicas_e_comportamentais'] = df['competencia_tecnicas_e_comportamentais'].fillna('').astype(str)
        df['principais_atividades'] = df['principais_atividades'].fillna('').astype(str)
        
        # Combinar textos da vaga
        df['texto_vaga_completo'] = (
            df['competencia_tecnicas_e_comportamentais'] + ' ' + 
            df['principais_atividades'] + ' ' + 
            df['titulo_vaga'].fillna('')
        )
        
        # TF-IDF Vectorizer otimizado
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True
        )
        
        # Combinar todos os textos para fit
        all_texts = df['cv_texto'].tolist() + df['texto_vaga_completo'].tolist()
        self.tfidf_vectorizer.fit(all_texts)
        
        # Vectorizar separadamente
        cv_vectors = self.tfidf_vectorizer.transform(df['cv_texto'])
        vaga_vectors = self.tfidf_vectorizer.transform(df['texto_vaga_completo'])
        
        # Calcular similaridades
        print("Calculando similaridades semanticas...")
        similarities = []
        for i in range(cv_vectors.shape[0]):
            sim = cosine_similarity(cv_vectors[i], vaga_vectors[i])[0][0]
            similarities.append(sim)
        
        df['match_score_semantico'] = similarities
        df['match_score_normalizado'] = (
            (df['match_score_semantico'] - df['match_score_semantico'].min()) / 
            (df['match_score_semantico'].max() - df['match_score_semantico'].min() + 1e-8)
        )
        
        # Features de matching basico (mantidas)
        df['skills_comuns'] = 0
        df['cv_skills_count'] = 0
        
        for skill in self.skills_tecnicas:
            cv_tem_skill = df['cv_texto'].str.contains(skill, case=False, na=False)
            vaga_tem_skill = df['texto_vaga_completo'].str.contains(skill, case=False, na=False)
            
            df['skills_comuns'] += (cv_tem_skill & vaga_tem_skill).astype(int)
            df['cv_skills_count'] += cv_tem_skill.astype(int)
        
        # Features de matching combinadas
        df['match_score_combinado'] = (
            0.6 * df['match_score_normalizado'] + 
            0.4 * (df['skills_comuns'] / (len(self.skills_tecnicas) + 1))
        )
        
        print(f"Match semantico medio: {df['match_score_semantico'].mean():.3f}")
        print(f"Skills comuns medio: {df['skills_comuns'].mean():.1f}")
        
        return df
    
    def criar_features_temporais_avancadas(self, df):
        """Cria features temporais avancadas"""
        print("\nCRIANDO FEATURES TEMPORAIS AVANCADAS...")
        
        # Converter datas
        df['data_candidatura'] = pd.to_datetime(df['data_candidatura'], errors='coerce')
        df['data_criacao'] = pd.to_datetime(df['data_criacao'], errors='coerce')
        df['data_requicisao'] = pd.to_datetime(df['data_requicisao'], errors='coerce')
        
        # Features de sazonalidade
        df['mes_candidatura'] = df['data_candidatura'].dt.month
        df['trimestre'] = df['data_candidatura'].dt.quarter
        df['dia_semana'] = df['data_candidatura'].dt.dayofweek
        df['semana_ano'] = df['data_candidatura'].dt.isocalendar().week
        
        # Features de tendencia
        data_min = df['data_candidatura'].min()
        if pd.notna(data_min):
            df['dias_desde_inicio'] = (df['data_candidatura'] - data_min).dt.days
            df['meses_desde_inicio'] = df['dias_desde_inicio'] // 30
        else:
            df['dias_desde_inicio'] = 0
            df['meses_desde_inicio'] = 0
        
        # Features sazonais especificas
        df['is_inicio_ano'] = (df['mes_candidatura'] <= 2).astype(int)
        df['is_meio_ano'] = ((df['mes_candidatura'] >= 6) & (df['mes_candidatura'] <= 8)).astype(int)
        df['is_fim_ano'] = (df['mes_candidatura'] >= 11).astype(int)
        df['is_segunda_feira'] = (df['dia_semana'] == 0).astype(int)
        df['is_sexta_feira'] = (df['dia_semana'] == 4).astype(int)
        
        # Tempo entre criacao do perfil e candidatura
        df['tempo_perfil_candidatura'] = (df['data_candidatura'] - df['data_criacao']).dt.days
        df['tempo_perfil_candidatura'] = df['tempo_perfil_candidatura'].fillna(0)
        
        # Tempo entre requisicao da vaga e candidatura
        df['tempo_vaga_candidatura'] = (df['data_candidatura'] - df['data_requicisao']).dt.days
        df['tempo_vaga_candidatura'] = df['tempo_vaga_candidatura'].fillna(0)
        
        print("Features temporais criadas")
        
        return df
    
    def criar_features_performance_historica(self, df):
        """Cria features de performance historica"""
        print("\nCRIANDO FEATURES DE PERFORMANCE HISTORICA...")
        
        # Performance por recrutador
        recrutador_stats = df.groupby('recrutador').agg({
            'cluster_target': ['count', lambda x: (x == 0).sum()]
        }).round(4)
        recrutador_stats.columns = ['total_candidatos', 'sucessos']
        recrutador_stats['taxa_sucesso_recrutador'] = recrutador_stats['sucessos'] / (recrutador_stats['total_candidatos'] + 1)
        recrutador_stats['experiencia_recrutador'] = np.log1p(recrutador_stats['total_candidatos'])
        
        # Mapear de volta
        df = df.merge(recrutador_stats.reset_index(), on='recrutador', how='left', suffixes=('', '_recrutador'))
        
        # Performance por modalidade
        modalidade_stats = df.groupby('modalidade').agg({
            'cluster_target': ['count', lambda x: (x == 0).sum()]
        }).round(4)
        modalidade_stats.columns = ['total_modalidade', 'sucessos_modalidade']
        modalidade_stats['taxa_sucesso_modalidade'] = modalidade_stats['sucessos_modalidade'] / (modalidade_stats['total_modalidade'] + 1)
        
        df = df.merge(modalidade_stats.reset_index(), on='modalidade', how='left', suffixes=('', '_modalidade'))
        
        # Performance por cliente
        if 'cliente' in df.columns:
            cliente_stats = df.groupby('cliente').agg({
                'cluster_target': ['count', lambda x: (x == 0).sum()]
            }).round(4)
            cliente_stats.columns = ['total_cliente', 'sucessos_cliente']
            cliente_stats['taxa_sucesso_cliente'] = cliente_stats['sucessos_cliente'] / (cliente_stats['total_cliente'] + 1)
            
            df = df.merge(cliente_stats.reset_index(), on='cliente', how='left', suffixes=('', '_cliente'))
        
        # Concorrencia por vaga
        concorrencia_stats = df.groupby('codigo_vaga').size().reset_index(name='concorrencia_vaga')
        df = df.merge(concorrencia_stats, on='codigo_vaga', how='left')
        df['log_concorrencia'] = np.log1p(df['concorrencia_vaga'])
        
        print("Features de performance historica criadas")
        
        return df
    
    def criar_features_qualidade_candidato(self, df):
        """Cria features de qualidade do candidato"""
        print("\nCRIANDO FEATURES DE QUALIDADE DO CANDIDATO...")
        
        # Qualidade do CV
        df['cv_length'] = df['cv_texto'].str.len()
        df['cv_words'] = df['cv_texto'].str.split().str.len()
        df['cv_sentences'] = df['cv_texto'].str.count(r'[.!?]+')
        df['cv_quality_score'] = (np.log1p(df['cv_length']) * np.log1p(df['cv_words'])) / (df['cv_sentences'] + 1)
        
        # Presenca de palavras-chave importantes
        keywords = ['experiencia', 'projeto', 'desenvolvimento', 'gestao', 'lideranca', 'equipe', 'resultado']
        for keyword in keywords:
            df[f'cv_tem_{keyword}'] = df['cv_texto'].str.contains(keyword, case=False, na=False).astype(int)
        
        # Diversidade de skills
        df['diversidade_skills'] = df['skills_comuns'] * np.log1p(df['cv_skills_count'])
        
        # Completude do perfil
        profile_cols = ['nome', 'email_secundario', 'telefone_celular', 'cv_texto', 'objetivo_profissional']
        existing_cols = [col for col in profile_cols if col in df.columns]
        if existing_cols:
            df['completude_perfil'] = df[existing_cols].notna().sum(axis=1) / len(existing_cols)
        else:
            df['completude_perfil'] = 0.5
        
        print("Features de qualidade do candidato criadas")
        
        return df
    
    def criar_features_interacao_complexas(self, df):
        """Cria features de interacao complexas"""
        print("\nCRIANDO FEATURES DE INTERACAO COMPLEXAS...")
        
        # Interacoes multiplicativas
        df['match_x_experiencia_recrutador'] = df['match_score_combinado'] * df['experiencia_recrutador']
        df['match_x_taxa_sucesso_recrutador'] = df['match_score_combinado'] * df['taxa_sucesso_recrutador']
        df['skills_x_taxa_sucesso_modalidade'] = df['skills_comuns'] * df['taxa_sucesso_modalidade']
        
        # Interacoes temporais
        df['match_x_mes'] = df['match_score_combinado'] * df['mes_candidatura']
        df['experiencia_x_trimestre'] = df['experiencia_recrutador'] * df['trimestre']
        
        # Interacoes de contexto
        df['concorrencia_x_match'] = df['concorrencia_vaga'] * df['match_score_combinado']
        df['qualidade_x_match'] = df['cv_quality_score'] * df['match_score_combinado']
        
        # Ratios importantes
        df['match_per_concorrencia'] = df['match_score_combinado'] / (df['concorrencia_vaga'] + 1)
        df['skills_per_experiencia'] = df['skills_comuns'] / (df['experiencia_recrutador'] + 1)
        
        print("Features de interacao complexas criadas")
        
        return df
    
    def encoding_categorico_inteligente(self, df):
        """Encoding categorico inteligente"""
        print("\nAPLICANDO ENCODING CATEGORICO INTELIGENTE...")
        
        # Colunas categoricas para encoding
        categorical_cols = [
            'modalidade', 'recrutador', 'sexo', 'estado_civil', 'nivel_academico',
            'nivel_ingles', 'nivel_profissional', 'tipo_contratacao', 'cliente',
            'requisitante', 'pais', 'estado_vaga', 'cidade_vaga'
        ]
        
        # Aplicar label encoding apenas nas colunas que existem
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Nao informado').astype(str)
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # Para dados novos, usar transform
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    except ValueError:
                        # Se houver categorias novas, refit
                        df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        print("Encoding categorico aplicado")
        
        return df
    
    def selecionar_features_finais(self, df):
        """Seleciona features finais para o modelo"""
        print("\nSELECIONANDO FEATURES FINAIS...")
        
        # Features principais (sempre incluir)
        features_principais = [
            'match_score_semantico', 'match_score_normalizado', 'match_score_combinado',
            'skills_comuns', 'cv_skills_count', 'diversidade_skills',
            'taxa_sucesso_recrutador', 'experiencia_recrutador', 'taxa_sucesso_modalidade',
            'concorrencia_vaga', 'log_concorrencia', 'cv_quality_score', 'completude_perfil'
        ]
        
        # Features temporais
        features_temporais = [
            'mes_candidatura', 'trimestre', 'dia_semana', 'is_inicio_ano', 'is_meio_ano', 'is_fim_ano',
            'tempo_perfil_candidatura', 'tempo_vaga_candidatura'
        ]
        
        # Features de interacao
        features_interacao = [
            'match_x_experiencia_recrutador', 'match_x_taxa_sucesso_recrutador',
            'skills_x_taxa_sucesso_modalidade', 'concorrencia_x_match',
            'match_per_concorrencia', 'skills_per_experiencia'
        ]
        
        # Features categoricas encoded
        features_categoricas = [col for col in df.columns if col.endswith('_encoded')]
        
        # Features de qualidade
        features_qualidade = [col for col in df.columns if col.startswith('cv_tem_')]
        
        # Combinar todas as features
        all_features = (features_principais + features_temporais + 
                       features_interacao + features_categoricas + features_qualidade)
        
        # Filtrar apenas features que existem no DataFrame
        features_existentes = [f for f in all_features if f in df.columns]
        
        print(f"Features selecionadas: {len(features_existentes)}")
        print(f"Principais categorias:")
        print(f"   - Matching: {len([f for f in features_existentes if 'match' in f])}")
        print(f"   - Temporais: {len([f for f in features_existentes if f in features_temporais])}")
        print(f"   - Performance: {len([f for f in features_existentes if 'taxa_sucesso' in f or 'experiencia' in f])}")
        print(f"   - Categoricas: {len(features_categoricas)}")
        print(f"   - Qualidade: {len(features_qualidade)}")
        
        return features_existentes
    
    def verificar_parquet_salvo(self, parquet_file):
        """Verifica o arquivo Parquet salvo - carrega, mostra primeiras linhas e schema"""
        print(f"\nVERIFICANDO ARQUIVO PARQUET SALVO...")
        print("-" * 50)
        
        try:
            # Carregar o DataFrame do arquivo Parquet
            df_carregado = pd.read_parquet(parquet_file, engine='pyarrow')
            
            print(f"Arquivo Parquet carregado com sucesso!")
            print(f"Dimensoes: {df_carregado.shape[0]:,} linhas x {df_carregado.shape[1]} colunas")
            
            # Mostrar as 5 primeiras linhas
            print(f"\nPRIMEIRAS 5 LINHAS DO DATASET CARREGADO:")
            print("-" * 50)
            print(df_carregado.head())
            
            # Capturar e mostrar o schema
            print(f"\nSCHEMA DO ARQUIVO PARQUET:")
            print("-" * 50)
            
            # Informacoes basicas do DataFrame
            print(f"Tipos de dados:")
            schema_info = df_carregado.dtypes.value_counts()
            for dtype, count in schema_info.items():
                print(f"  - {dtype}: {count} colunas")
            
            # Schema detalhado (primeiras 10 colunas para nao poluir)
            print(f"\nDetalhes das primeiras 10 colunas:")
            for i, (col, dtype) in enumerate(df_carregado.dtypes.head(10).items()):
                print(f"  {i+1:2d}. {col:<30} | {str(dtype):<15}")
            
            if len(df_carregado.columns) > 10:
                print(f"  ... e mais {len(df_carregado.columns) - 10} colunas")
            
            # Estatisticas da coluna target se existir
            if 'cluster_target' in df_carregado.columns:
                print(f"\nDISTRIBUICAO DA VARIAVEL TARGET:")
                target_dist = df_carregado['cluster_target'].value_counts().sort_index()
                total = len(df_carregado)
                
                clusters_nomes = {
                    0: "CANDIDATOS DE SUCESSO",
                    1: "CANDIDATOS QUE SAIRAM DO PROCESSO", 
                    2: "CANDIDATOS EM PROCESSO"
                }
                
                for cluster_id, count in target_dist.items():
                    pct = (count / total) * 100
                    nome = clusters_nomes.get(cluster_id, f"CLUSTER {cluster_id}")
                    print(f"  Cluster {cluster_id} - {nome}: {count:,} ({pct:.1f}%)")
            
            # Informacoes de memoria
            memory_usage = df_carregado.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"\nUso de memoria: {memory_usage:.2f} MB")
            
            # Tamanho do arquivo
            file_size = os.path.getsize(parquet_file) / 1024 / 1024
            print(f"Tamanho do arquivo: {file_size:.2f} MB")
            
            return df_carregado
            
        except Exception as e:
            print(f"Erro ao verificar arquivo Parquet: {str(e)}")
            return None
    
    def carregar_dados_processados(self):
        """Carrega dados processados do arquivo Parquet"""
        print("CARREGANDO DADOS PROCESSADOS...")
        
        parquet_file = os.path.join(self.artifacts_dir, 'processed_data_3_clusters_melhorado.parquet')
        artifacts_file = os.path.join(self.artifacts_dir, 'preprocessing_artifacts_3_clusters_melhorado.pkl')
        
        try:
            # Carregar DataFrame do Parquet
            df = pd.read_parquet(parquet_file, engine='pyarrow')
            print(f"Dataset carregado: {df.shape}")
            
            # Carregar artefatos do pickle
            with open(artifacts_file, 'rb') as f:
                artifacts = pickle.load(f)
            
            # Separar features e target
            feature_names = artifacts['feature_names']
            X = df[feature_names]
            y = df['cluster_target']
            
            # Restaurar objetos de preprocessamento
            self.label_encoders = artifacts['label_encoders']
            self.scaler = artifacts['scaler']
            self.tfidf_vectorizer = artifacts['tfidf_vectorizer']
            self.clusters_mapping = artifacts['clusters_mapping']
            self.skills_tecnicas = artifacts['skills_tecnicas']
            
            print("Artefatos de preprocessamento carregados")
            
            return X, y, feature_names, artifacts
            
        except FileNotFoundError as e:
            print(f"Arquivo nao encontrado: {str(e)}")
            return None, None, None, None
        except Exception as e:
            print(f"Erro ao carregar dados: {str(e)}")
            return None, None, None, None
    
    def comparar_formatos_salvamento(self, df):
        """Compara tamanhos de arquivos entre Parquet e Pickle"""
        print("\nCOMPARANDO FORMATOS DE SALVAMENTO...")
        
        # Salvar temporariamente em ambos os formatos
        temp_parquet = os.path.join(self.artifacts_dir, 'temp_comparison.parquet')
        temp_pickle = os.path.join(self.artifacts_dir, 'temp_comparison.pkl')
        
        try:
            # Salvar em Parquet
            start_time = time.time()
            df.to_parquet(temp_parquet, engine='pyarrow', compression='snappy')
            parquet_time = time.time() - start_time
            parquet_size = os.path.getsize(temp_parquet) / 1024 / 1024
            
            # Salvar em Pickle
            start_time = time.time()
            with open(temp_pickle, 'wb') as f:
                pickle.dump(df, f)
            pickle_time = time.time() - start_time
            pickle_size = os.path.getsize(temp_pickle) / 1024 / 1024
            
            # Comparar resultados
            print(f"Parquet (.parquet):")
            print(f"  - Tamanho: {parquet_size:.2f} MB")
            print(f"  - Tempo de escrita: {parquet_time:.3f} segundos")
            print(f"  - Compressao: Snappy")
            
            print(f"\nPickle (.pkl):")
            print(f"  - Tamanho: {pickle_size:.2f} MB") 
            print(f"  - Tempo de escrita: {pickle_time:.3f} segundos")
            print(f"  - Compressao: Nenhuma")
            
            print(f"\nEconomia de espaco com Parquet: {((pickle_size - parquet_size) / pickle_size * 100):.1f}%")
            
            # Limpar arquivos temporarios
            os.remove(temp_parquet)
            os.remove(temp_pickle)
            
        except Exception as e:
            print(f"Erro na comparacao: {str(e)}")
    
    def executar_processamento_completo(self):
        """Executa todo o pipeline de processamento"""
        print("INICIANDO PROCESSAMENTO MELHORADO FINAL - 3 CLUSTERS")
        print("=" * 70)
        
        # 1. Carregar e preparar dados
        df = self.load_and_prepare_data()
        
        # 2. Criar clusters target
        df = self.criar_clusters_target(df)
        
        # 3. Limpeza basica
        df = self.limpeza_basica_melhorada(df)
        
        # 4. Features de matching semantico
        df = self.criar_features_matching_semantico(df)
        
        # 5. Features temporais avancadas
        df = self.criar_features_temporais_avancadas(df)
        
        # 6. Features de performance historica
        df = self.criar_features_performance_historica(df)
        
        # 7. Features de qualidade do candidato
        df = self.criar_features_qualidade_candidato(df)
        
        # 8. Features de interacao complexas
        df = self.criar_features_interacao_complexas(df)
        
        # 9. Encoding categorico
        df = self.encoding_categorico_inteligente(df)
        
        # 10. Selecionar features finais
        feature_names = self.selecionar_features_finais(df)
        
        # 11. Preparar dados finais
        X = df[feature_names].fillna(0)
        y = df['cluster_target']
        
        # 12. Normalizar features numericas
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
        
        print(f"\nDataset final: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Distribuicao target: {y.value_counts().sort_index().to_dict()}")
        
        # 13. Salvar dados processados
        print("\nSalvando dados processados...")
        
        # Adicionar a coluna target de volta ao DataFrame para salvar completo
        df_final = X.copy()
        df_final['cluster_target'] = y
        
        # Salvar dataset principal em PARQUET
        parquet_file = os.path.join(self.artifacts_dir, 'processed_data_3_clusters_melhorado.parquet')
        df_final.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
        print(f"Dataset principal salvo em Parquet: {parquet_file}")
        
        # Salvar artefatos de preprocessamento (mantém pickle para objetos Python complexos)
        artifacts = {
            'feature_names': feature_names,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'clusters_mapping': self.clusters_mapping,
            'skills_tecnicas': self.skills_tecnicas
        }
        
        with open(os.path.join(self.artifacts_dir, 'preprocessing_artifacts_3_clusters_melhorado.pkl'), 'wb') as f:
            pickle.dump(artifacts, f)
        
        print("Artefatos de preprocessamento salvos em pickle")
        
        # Verificar arquivo Parquet salvo
        self.verificar_parquet_salvo(parquet_file)
        
        # Opcional: comparar formatos
        # self.comparar_formatos_salvamento(df_final)
        
        print("\nPROCESSAMENTO MELHORADO FINAL CONCLUIDO!")
        print("=" * 70)
        
        return X, y, feature_names

def main():
    """Funcao principal"""
    processor = ProcessadorDados3ClustersMelhorado()
    X, y, feature_names = processor.executar_processamento_completo()
    
    print(f"\nResumo final:")
    print(f"   - Registros: {len(X):,}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Clusters: {len(y.unique())}")
    print(f"   - Taxa de sucesso (Cluster 0): {(y == 0).mean():.1%}")

if __name__ == "__main__":
    main()
