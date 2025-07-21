# Use uma imagem base do Python
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie o arquivo de dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie a pasta do seu feature repo para o diretório de trabalho
COPY ./feature_repo/ .

# A UI do Feast roda na porta 8888 por padrão
EXPOSE 8888

# Comando para iniciar a UI do Feast quando o container iniciar
# O host 0.0.0.0 é essencial para que o Render possa acessar o serviço
CMD ["feast", "ui", "-h", "0.0.0.0", "-p", "8888"]
