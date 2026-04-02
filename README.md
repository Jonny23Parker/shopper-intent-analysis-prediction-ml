# Shopper Intent Analysis & Purchase Prediction

Projeto de análise comportamental e predição de intenção de compra em ambiente de e-commerce utilizando dados de clickstream em larga escala.

---

## Objetivo do Projeto

Este projeto teve como objetivo analisar o comportamento de navegação dos usuários em um e-commerce e prever a probabilidade de conversão em compra por sessão.

A abordagem foi construída em **duas etapas**:

1. **prototipação com amostragem de 100 mil linhas**
2. **escalabilidade e validação na base completa com 5,4 milhões de linhas**

Essa evolução foi proposital para validar hipóteses, testar performance e posteriormente aplicar o modelo em ambiente de maior volume.

---

## Dataset

Dataset oficial utilizado:

**Shopper Intent Prediction Dataset**  
Fonte oficial:  
https://staticassets.coveo.com/ai-labs/shopper_intent_prediction.zip

Após extração, utilizar o arquivo:

```text
release_10_23_2020.csv
```

---

## Etapa 1 — Prototipação com Amostragem (100 mil linhas)

A primeira fase do projeto foi realizada com uma amostra controlada de:

```text
100.000 linhas
```

Objetivos desta etapa:

- validar estrutura dos dados
- entender colunas críticas
- testar engenharia de features
- validar primeira versão do modelo
- medir baseline de performance

### Resultados iniciais (100k)

### Dataset amostral

| Métrica | Valor |
|---|---:|
| Linhas | 100.000 |
| Sessões | 7.715 |
| Compras | 178 |

### Performance inicial

| Métrica | Resultado |
|---|---:|
| Accuracy | 93% |
| ROC AUC | 0.8737 |
| Recall compra | 47% |

Essa fase foi essencial para identificar:

- forte desbalanceamento
- importância do carrinho
- necessidade de SMOTE


Arquivo usado na amostragem:

```text
eda 100K Amostra.py
```


---

## Etapa 2 — Escalabilidade com Dataset Completo

Após validação do pipeline inicial, o modelo foi executado com a base completa:

```text
5.433.611 linhas
```


Arquivo usado na amostragem:

```text
eda.py
```




Essa etapa teve foco em:

- escalabilidade
- consumo de memória
- validação em volume real
- robustez do modelo

---

## Volume Processado

| Métrica | Valor |
|---|---:|
| Total de linhas | 5.433.611 |
| Pageviews | 4.565.253 |
| Events | 868.358 |
| Compras | 9.926 |

---

## Abordagem de Processamento

O projeto foi executado em duas estratégias:

### 1. leitura completa em memória

```python
df = pd.read_csv(file_path)
```

Essa abordagem foi possível devido à disponibilidade de memória RAM suficiente.

---

### 2. estratégia preparada para grandes volumes (chunk processing)

Também foi estudada abordagem de escalabilidade com chunks para ambientes produtivos:

```python
for chunk in pd.read_csv(file_path, chunksize=500000):
    process(chunk)
```

Essa abordagem é recomendada para:

- pipelines ETL
- Data Engineering
- Airflow / Spark migration
- datasets acima de memória disponível

> Observação: esta abordagem foi planejada como melhoria de arquitetura e escalabilidade.

---

## Exploração dos Dados (EDA)

### Distribuição de eventos

| Evento | Quantidade |
|---|---:|
| pageview | 4.565.253 |
| event | 868.358 |

---

### Ações do produto

| Ação | Quantidade |
|---|---:|
| detail | 1.640.190 |
| add | 743.363 |
| click | 69.831 |
| remove | 51.512 |
| purchase | 9.926 |

---

### Valores nulos

| Coluna | Missing |
|---|---:|
| product_action | 2.918.789 |
| product_skus_hash | 3.547.557 |

---

## Engenharia de Features

Os dados foram agregados por sessão.

### Features criadas

- `num_events`
- `num_detail`
- `num_add`
- `num_remove`
- `session_duration`

---

## Variável Alvo

```python
made_purchase
```

| Classe | Quantidade |
|---|---:|
| Não comprou | 434.428 |
| Comprou | 9.232 |

---

## Tratamento de Desbalanceamento

Foi aplicado:

```python
SMOTE
```

Somente no conjunto de treino.

Objetivo:

- aumentar sensibilidade para classe minoritária
- melhorar recall de compradores

---

## Modelagem

Modelo utilizado:

```python
RandomForestClassifier
```

Bibliotecas:

- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib

---

## Resultados Finais — Base Completa

### Classification Report

| Classe | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Não compra | 0.99 | 0.93 | 0.96 |
| Compra | 0.19 | **0.77** | 0.31 |

---

## Métricas Finais

| Métrica | Resultado |
|---|---:|
| Accuracy | 93% |
| ROC AUC | **0.9262** |
| Recall compradores | **77%** |

---

## Evolução do Modelo

| Etapa | Recall | ROC AUC |
|---|---:|---:|
| 100k amostra | 47% | 0.8737 |
| 5.4M full | **77%** | **0.9262** |

---

## Principal Insight de Negócio

A feature mais importante foi:

```text
num_add
```

Ou seja:

**adição ao carrinho é o principal indicador de intenção de compra**

Outras features relevantes:

- duração da sessão
- volume de interações
- visualizações de detalhe

---

## Estrutura do Projeto

```text
shopper_intent_project/
│
├── data/
│   └── release_10_23_2020.csv
│
├── notebooks/
│   └── eda.py
│
├── images/
│   └── feature_importance.png
│
└── README.md
```

---

## Próximas Melhorias (Sugestões)

Itens abaixo são sugestões futuras e **não foram implementados ainda**:

- pipeline ETL com Airflow
- processamento por chunks automatizado
- ingestão incremental
- feature store
- deploy via API
- dashboard executivo Power BI
- monitoramento do modelo

---

## Stack

- Python
- Pandas
- Scikit-learn
- SMOTE
- Random Forest
- Matplotlib
- VS Code

---

## Autor

**Jonatas de Siqueira Bitencourt Cursino**  
Data Analyst | Machine Learning | Data Engineering
