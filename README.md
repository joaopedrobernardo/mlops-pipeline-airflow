# 🚀 MLOps Pipeline: Airflow + Elasticsearch + LLM/RAG + Grafana

![MLOps](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Airflow](https://img.shields.io/badge/Orchestration-Apache%20Airflow-orange)
![Docker](https://img.shields.io/badge/Containerization-Docker-blue)
![Elasticsearch](https://img.shields.io/badge/Search-Elasticsearch-green)
![Grafana](https://img.shields.io/badge/Monitoring-Grafana-red)
![LLM](https://img.shields.io/badge/AI-LLM%20%2B%20RAG-purple)
![Python](https://img.shields.io/badge/Language-Python-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

> **A complete, production-grade MLOps pipeline for operationalizing and monitoring Generative AI with Large Language Models (LLM) and Retrieval-Augmented Generation (RAG).**

This project demonstrates a full end-to-end MLOps architecture, from data ingestion to AI-powered question answering, with monitoring and orchestration. Built as part of the Machine Learning 3 course at [UNIFEI](https://unifei.edu.br/).

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [What This Project Demonstrates](#-what-this-project-demonstrates)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Services & Ports](#-services--ports)
- [Pipeline Details](#-pipeline-details)
- [Feature Store Module](#-feature-store-module)
- [Screenshots](#-screenshots)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCKER NETWORK                               │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   PostgreSQL │    │     Redis    │    │   Elasticsearch 8.x  │  │
│  │   (Metadata) │    │   (Broker)   │    │   (Vector Store)     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │               │
│  ┌──────┴───────────────────┴───────────────────────┴───────────┐  │
│  │                    APACHE AIRFLOW 2.10                        │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────────┐  │  │
│  │  │Webserver│  │Scheduler │  │ Worker  │  │   Triggerer   │  │  │
│  │  │  :8080  │  │          │  │ (Celery)│  │               │  │  │
│  │  └─────────┘  └──────────┘  └─────────┘  └───────────────┘  │  │
│  │                                                               │  │
│  │  DAG: DSA_Carrega_Dados_RAG                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │ Cria     │→ │ Carrega  │→ │ Carrega  │→ │ Cria Índice │  │  │
│  │  │ Tabela   │  │ JSON     │  │ CSV      │  │ Elastic     │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │     STREAMLIT APP :8501     │  │    GRAFANA DASHBOARD :3000  │  │
│  │                             │  │                             │  │
│  │  ┌───────────────────────┐  │  │  ┌───────────────────────┐  │  │
│  │  │   LLM + RAG Engine    │  │  │  │  Monitoring & Alerts  │  │  │
│  │  │   (HuggingFace API)   │  │  │  │  (Response Metrics)   │  │  │
│  │  └───────────────────────┘  │  │  └───────────────────────┘  │  │
│  │  ┌───────────────────────┐  │  │                             │  │
│  │  │  User Feedback System │  │  │                             │  │
│  │  └───────────────────────┘  │  │                             │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What This Project Demonstrates

### MLOps & Orchestration
- **Apache Airflow** DAGs for automated data pipeline orchestration
- **Docker Compose** multi-container architecture with service dependencies
- **CeleryExecutor** pattern with Redis broker and PostgreSQL backend
- Scheduled daily pipeline execution with retry logic

### AI/LLM Integration
- **Retrieval-Augmented Generation (RAG)** with HuggingFace LLM models
- **Elasticsearch** as vector store for semantic document search
- **Streamlit** web interface for interactive Q&A with AI
- User feedback collection and evaluation metrics (Hit Rate, MRR)

### Data Engineering
- **Feature Store** implementation with feature engineering pipeline
- **ETL processes**: JSONL and CSV data ingestion into PostgreSQL
- **Data versioning** with MD5 hash-based document IDs
- **RandomForest** model training with serialized model artifacts

### Monitoring & Observability
- **Grafana** dashboards for real-time AI response monitoring
- **Airflow UI** for pipeline execution tracking and logs
- Response time tracking and quality metrics

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Orchestration** | Apache Airflow 2.10.2 |
| **Containerization** | Docker, Docker Compose |
| **Database** | PostgreSQL 13, Redis 7.2 |
| **Search/Vector Store** | Elasticsearch 8.15.1 |
| **AI/LLM** | HuggingFace Transformers, RAG |
| **Web Framework** | Streamlit |
| **Monitoring** | Grafana |
| **ML Training** | scikit-learn, RandomForest |
| **Language** | Python 3.11+ |

---

## 📁 Project Structure

```
mlops-pipeline-airflow/
│
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
│
├── docker-compose.yaml                # Main orchestration file
├── .env                               # Environment variables (create from .env.example)
│
├── 01_airflow_pipeline/               # Airflow DAGs and modules
│   ├── Dockerfile                     # Custom Airflow image
│   ├── config/
│   │   └── airflow.cfg                # Airflow configuration
│   ├── dags/
│   │   ├── dsapipeline.py             # Main DAG definition
│   │   └── modulodsadados/            # Data loading modules
│   │       ├── __init__.py
│   │       ├── dsaconnection.py       # PostgreSQL connection
│   │       ├── dsa_get_dados.py       # Data extraction
│   │       └── dsa_carrega_dados.py   # Data loading functions
│   └── dados/                         # Source data files
│       ├── dataset1.jsonl             # JSONL legal Q&A data
│       └── dataset2.csv               # CSV case data
│
├── 02_streamlit_app/                  # LLM/RAG Web Application
│   ├── Dockerfile                     # App container image
│   ├── requirements.txt               # Python dependencies
│   ├── appdsa.py                      # Main Streamlit application
│   └── app/                           # Application modules
│       ├── __init__.py
│       ├── dsaconnection.py           # DB connection
│       ├── dsaelasticSearch.py        # Elasticsearch client
│       ├── dsallm.py                  # LLM query functions
│       └── dsaevaluation.py           # Evaluation metrics
│
├── 03_feature_store/                  # Feature Store + ML Pipeline
│   ├── requirements.txt               # Python dependencies
│   ├── projeto3-main.py               # Main execution script
│   ├── projeto3-app.py                # Flask API for predictions
│   ├── projeto3-cliente.py            # Test client
│   ├── dsa_dados/                     # Data directory
│   │   ├── dados_brutos.csv           # Raw data
│   │   ├── feature_store.csv          # Processed features
│   │   └── teste_features.csv         # Test features
│   ├── dsa_feature_store/             # Feature engineering module
│   │   ├── __init__.py
│   │   ├── feature_engineering.py     # Feature creation logic
│   │   └── feature_store.py           # Feature store class
│   ├── dsa_ml_pipeline/               # ML training & inference
│   │   ├── __init__.py
│   │   ├── model_training.py          # RandomForest training
│   │   ├── model_inference.py         # Prediction inference
│   │   ├── modelo_dsa.pkl             # Serialized model
│   │   └── scaler.pkl                 # Feature scaler
│   └── dsa_testes/                    # Unit tests
│       └── test_feature_store.py      # Feature store tests
│
├── 04_grafana_dashboard/              # Grafana configuration
│   └── dashboard.json                 # Dashboard template
│
└── docs/                              # Additional documentation
    ├── architecture.md                # Detailed architecture docs
    └── setup_guide.md                 # Step-by-step setup guide
```

---

## 📋 Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-compose/) (Windows/Mac/Linux)
- [HuggingFace Account](https://huggingface.co/) + API Token (free)
- 8GB+ RAM available for Docker
- Git

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/mlops-pipeline-airflow.git
cd mlops-pipeline-airflow
```

### 2. Configure Environment Variables
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your HuggingFace token
HUGGINGFACE_KEY=hf_YOUR_TOKEN_HERE
```

### 3. Start All Services
```bash
docker-compose up --build -d
```

### 4. Configure Elasticsearch Hostname
```bash
# Find the Elasticsearch container ID
docker ps | grep elasticsearch

# Update the hostname in these files:
# - 01_airflow_pipeline/dags/modulodsadados/dsa_carrega_dados.py (line 185)
# - 02_streamlit_app/app/dsaelasticSearch.py (line 11)
```

### 5. Access the Services

| Service | URL | Credentials |
|---|---|---|
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **Streamlit App** | http://localhost:8501 | - |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| **Elasticsearch** | http://localhost:9200 | - |

### 6. Activate the DAG
1. Go to Airflow UI → `DSA_Carrega_Dados_RAG`
2. Toggle the DAG to **ON**
3. Wait for the pipeline to complete

### 7. Ask Questions
1. Go to Streamlit App: http://localhost:8501
2. Try these example questions:
   - *"Can the landlord avoid liability for breaching this obligation if the state of disrepair is caused by the tenant's actions?"*
   - *"Why did the plaintiff wait seven months to file an appeal?"*
   - *"Can you provide more details on the clarification provided in Note 1?"*

---

## 🔌 Services & Ports

| Service | Port | Description |
|---|---|---|
| Airflow Webserver | 8080 | DAG management UI |
| Airflow Scheduler | - | Task scheduling |
| Airflow Worker | - | Celery task execution |
| PostgreSQL | 5432 | Airflow metadata + app data |
| Redis | 6379 | Celery message broker |
| Elasticsearch | 9200, 9300 | Vector search index |
| Streamlit App | 8501 | LLM/RAG web interface |
| Grafana | 3000 | Monitoring dashboard |

---

## 📊 Pipeline Details

### Airflow DAG: `DSA_Carrega_Dados_RAG`

```
┌─────────────────┐
│  dsa_cria_tabela │  → Creates PostgreSQL table
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│dsa_insere_dados │  → Loads JSONL data (25 records)
│     _json       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│dsa_insere_dados │  → Loads CSV data (25 records)
│      _csv       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ dsa_cria_indice │  → Creates Elasticsearch index
└─────────────────┘     with all documents
```

**Schedule:** Daily at midnight (`0 0 * * *`)
**Retries:** 1 with 1-hour delay

### Streamlit RAG Flow

```
User Question → Elasticsearch (context retrieval) → HuggingFace LLM → Answer
                     ↓                                      ↓
              Top-k documents                        Response + Score
                                                          ↓
                                                   User Feedback
                                                   (Satisfied/Not)
```

---

## 🗂️ Feature Store Module

A standalone ML pipeline demonstrating feature engineering best practices:

```bash
cd 03_feature_store
pip install -r requirements.txt

# Run unit tests
python -m unittest discover -s dsa_testes

# Execute full pipeline
python projeto3-main.py

# Start prediction API
python projeto3-app.py

# Test the API (in another terminal)
python projeto3-cliente.py
```

**Features:**
- Automated feature engineering with `StandardScaler`
- Feature persistence in CSV format
- Model serialization with `joblib`
- REST API for real-time predictions
- Unit tests for feature validation

---

## 📸 Screenshots

### Airflow DAG Execution
![Airflow DAG](docs/images/airflow_dag.png)

### Streamlit RAG Interface
![Streamlit App](docs/images/streamlit_app.png)

### Grafana Monitoring
![Grafana Dashboard](docs/images/grafana_dashboard.png)

---

## 🎓 Key Learnings

Through this project, I gained hands-on experience with:

1. **MLOps Best Practices**
   - Containerized ML services with Docker
   - Orchestration with Apache Airflow
   - Pipeline monitoring and alerting

2. **LLM Operationalization**
   - RAG architecture for domain-specific Q&A
   - Elasticsearch as a vector database
   - HuggingFace model integration

3. **Data Engineering**
   - Feature store design patterns
   - ETL pipeline development
   - Data versioning and lineage

4. **Production Considerations**
   - Service health checks and dependencies
   - Environment variable management
   - Scalable container architecture

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**João Pedro Bernardo de Paula**

- 🎓 Computer Engineering Student at UNIFEI (Brazil)
- 💼 Data Science & Machine Learning Enthusiast
- 🌍 Seeking opportunities in the USA/Europe

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/jpbernardodepaula)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github)](https://github.com/jpbernardodepaula)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail)](mailto:jpbernardodepaula@gmail.com)

---

> **Note:** This project was developed as part of the Machine Learning 3 course at UNIFEI. The original project structure and some code patterns follow the Data Science Academy (DSA) curriculum, with significant modifications and enhancements for production readiness.
