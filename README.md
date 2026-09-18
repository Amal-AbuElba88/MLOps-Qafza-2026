# MLOps Engineering Training Program

## Qafza Tech — 12-Week Applied Program

Welcome to the **MLOps Engineering** training program by **[Qafza Tech](https://www.qafzatech.com)** . This hands-on journey takes you from local development to production-grade ML systems, using the Olist Brazilian E-Commerce dataset as our real-world case study.

---

## About Qafza Tech

**Qafza Tech** is a leading data and AI academy in the Middle East, trusted by **5,000+ trainees** and **20+ companies** including STC, Saudi Aramco, SDAIA, and Microsoft. With a **4.8/5 trainee rating**, Qafza delivers practical training that bridges the gap between learning and employment.

> *"From zero to job-ready — real projects, certified training, and career launch support."*

---

## Program Overview

### The Challenge
Predict whether an e-commerce order will be **delivered late or on time** — a critical business problem that impacts customer satisfaction and operational efficiency.

### The Journey
Over **12 intensive weeks**, you'll build an end-to-end ML system:

```mermaid
graph LR
    A[Weeks 1-2<br/>Local Foundations] --> B[Weeks 3-4<br/>Production APIs & Docker]
    B --> C[Weeks 5-7<br/>Data Pipelines & Versioning]
    C --> D[Weeks 8-10<br/>Distributed Training & Feature Store]
    D --> E[Weeks 11-12<br/>Monitoring & Automation]
    E --> F[Capstone<br/>End-to-End ML System]
```

---

## Weekly Curriculum

| Week | Phase | Dates | Topic | Focus Area |
|------|-------|-------|-------|------------|
| **Week 1** | Local Foundations | 20–27 Jul 2026 | **Leakage-proof ML Pipeline** | Build robust pipelines that prevent data leakage, ensuring model integrity from day one. |
| **Week 2** | Local Foundations | 27 Jul – 3 Aug 2026 | **Deep Learning Pipeline** | Extend your pipeline with neural network architectures using PyTorch/TensorFlow. |
| **Week 3** | Production APIs | 3–10 Aug 2026 | **Production API** | Create FastAPI/Flask endpoints to serve your model as a RESTful service. |
| **Week 4** | Containerization | 10–17 Aug 2026 | **Docker** | Containerize your application, ensuring consistency across development and production. |
| **Week 5** | Data Pipelines | 17–24 Aug 2026 | **ETL Pipeline** | Design robust Extract-Transform-Load workflows using tools like Airflow or Prefect. |
| **Week 6** | Data Versioning | 24–31 Aug 2026 | **Versioning** | Implement DVC to version datasets, ensuring reproducibility and collaboration. |
| **Week 7** | Experiment Tracking | 31 Aug – 7 Sep 2026 | **MLflow** | Track experiments, log parameters, metrics, and artifacts for complete reproducibility. |
| **Week 8** | Distributed Training | 7–14 Sep 2026 | **Distributed ML** | Scale training across multiple GPUs/nodes using Ray, Horovod, or PyTorch Distributed. |
| **Week 9** | Feature Store | 14–21 Sep 2026 | **Feature Management** | Build and manage a feature store (Feast/Hopsworks) for online/offline serving. |
| **Week 10** | Monitoring | 21–28 Sep 2026 | **Monitoring** | Implement model and data drift monitoring using Prometheus/Grafana or Evidently AI. |
| **Week 11** | Continuous Retraining | 28 Sep – 5 Oct 2026 | **Automation** | Build automated retraining pipelines triggered by data drift or schedule. |
| **Week 12** | Infrastructure as Code | 5–12 Oct 2026 | **Infrastructure** | Define cloud infrastructure using Terraform/Pulumi for reproducible deployment. |
| **Final** | Capstone | 12–19 Oct 2026 | **End-to-End ML System** | Integrate all components into a complete, production-ready ML system. |

---

## Technology Stack

### Core Technologies by Phase

| Phase | Technologies |
|-------|-------------|
| **Foundations** | Python, Pandas, NumPy, Scikit-learn, Jupyter |
<!-- | **Deep Learning** | PyTorch, TensorFlow, Keras |
| **APIs & Deployment** | FastAPI, Flask, Gunicorn, NGINX | -->
| **Containerization** | Docker, Docker Compose |
<!-- | **Data Pipelines** | Apache Airflow, Prefect, Dagster |
| **Versioning** | DVC (Data Version Control), Git LFS |
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Distributed Training** | PyTorch Distributed, Ray, Horovod |
| **Feature Store** | Feast, Hopsworks, Featureform |
| **Monitoring** | Prometheus, Grafana, Evidently AI, WhyLabs | -->
| **CI/CD** | GitHub Actions |
<!-- | **Infrastructure** | Terraform, Pulumi, AWS CDK | -->
<!-- 
### Python Ecosystem

```python
# Core
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0

# Deep Learning
torch>=2.0.0
tensorflow>=2.13.0

# MLOps
mlflow>=2.5.0
dvc>=3.0.0
prefect>=2.0.0
feast>=0.33.0
evidently>=0.3.0

# API & Web
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Infrastructure
boto3>=1.28.0
terraform>=1.5.0

# Testing & Quality
pytest>=7.4.0
black>=23.0.0
flake8>=6.1.0
``` -->

---

## Project: Olist Delivery Prediction

### Dataset Overview

The **Olist Brazilian E-Commerce Dataset** contains real marketplace data:
<!-- 
```mermaid
erDiagram
    CUSTOMERS ||--o{ ORDERS : places
    ORDERS ||--o{ ORDER_ITEMS : contains
    ORDERS ||--o{ ORDER_PAYMENTS : has
    ORDERS ||--o{ ORDER_REVIEWS : receives
    ORDER_ITEMS }o--|| PRODUCTS : references
    ORDER_ITEMS }o--|| SELLERS : sold_by
    PRODUCTS }o--|| CATEGORY : belongs_to
```

### Key Statistics

| Table | Size | Purpose |
|-------|------|---------|
| orders | 100,000+ | Main transaction records |
| customers | 100,000+ | Customer profiles |
| order_items | 110,000+ | Line items per order |
| products | 30,000+ | Product catalog |
| sellers | 3,000+ | Seller information |
| order_payments | 100,000+ | Payment transactions |
| order_reviews | 100,000+ | Customer feedback |
| geolocation | 1,000,000+ | Location coordinates |

### Business Impact

**Customer Satisfaction** — Reduce late deliveries and improve customer experience  
**Operational Efficiency** — Proactively address logistics bottlenecks  
**Revenue Protection** — Minimize refunds and chargebacks due to delivery issues  
**Scalable Solution** — Framework applicable to other e-commerce platforms

---

## Key Learning Outcomes

### Technical Skills
- Build **production-ready ML pipelines** with proper data validation
- Implement **API services** for model serving
- **Containerize** applications using Docker
- Design **ETL/ELT data pipelines** using modern orchestration tools
- Version **data, code, and models** for full reproducibility
- Track **experiments** to compare and select optimal models
- Scale training with **distributed computing**
- Build and manage **feature stores**
- Implement **monitoring** for data and model drift
- Automate **retraining pipelines**
- Deploy infrastructure using **Infrastructure as Code (IaC)**

### Professional Competencies
- **System Design** — Architect complete ML systems
- **Data Engineering** — Build robust data pipelines
- **Model Development** — Create and optimize ML models
- **DevOps Practices** — Containerization, CI/CD, and monitoring
- **Documentation** — Create clear technical documentation
- **Collaboration** — Work with version control and team workflows
- **Problem Solving** — Address real-world business challenges

---
-->
## Capstone Project

### Final Deliverable: End-to-End ML System

Integrate all 12 weeks of learning into a complete solution:

```mermaid
graph TD
    A[Data Ingestion] --> B[Feature Engineering]
    B --> C[Model Training]
    C --> D[Model Registry]
    D --> E[Model Deployment]
    E --> F[Monitoring & Retraining]
    F --> C
```

### Capstone Requirements

| Component | Description |
|-----------|-------------|
| **Data Pipeline** | Automated ETL from raw CSV to feature store |
| **Feature Store** | Online/offline serving with Feast |
| **Model Training** | Automated retraining pipeline |
| **Experiment Tracking** | MLflow with full parameter logging |
| **Model Registry** | Versioned model artifacts |
| **API Service** | FastAPI with prediction endpoints |
| **Containerization** | Dockerized application |
| **CI/CD** | GitHub Actions pipeline |
| **Monitoring** | Data drift + model performance dashboards |
| **Infrastructure** | Terraform-defined deployment | 

---

### Program Strengths

- **Structured Curriculum** — Week-by-week progression
- **Expert Instructors** — Industry professionals with real experience
- **Job-Ready Skills** — Focus on practical, in-demand tools
- **Certification** — Recognized credential upon completion
- **Community** — Join 5,000+ alumni network
- **Employment Support** — Connection to hiring partners

---

## Getting Started

### Prerequisites

| Skill | Level |
|-------|-------|
| Python | Intermediate |
| SQL | Basic |
| Git | Basic |
| Linux/Command Line | Basic |

### Hardware Requirements

- 8GB+ RAM (16GB recommended)
- 50GB free disk space
- GPU optional (cloud resources provided)

## Program Timeline at a Glance

```
Week 1-2  ████████░░░░░░░░  Local Foundations (ML Pipeline + Deep Learning)
Week 3-4  ░░████████░░░░░░  Production APIs & Containerization
Week 5-6  ░░░░░████████░░░░  Data Pipelines & Versioning
Week 7-8  ░░░░░░░░████████  Experiment Tracking & Distributed Training
Week 9-10 ░░░░░░░░░░░██████  Feature Store & Monitoring
Week 11-12░░░░░░░░░░░░░░████  Retraining & Infrastructure
Capstone  ░░░░░░░░░░░░░░░░██████  End-to-End ML System
```

---

## License

This training program is proprietary to Qafza Tech. Materials are provided for educational purposes only.



---

## Task 03: From Notebooks to Production

This repository turns the fitted Task 2 pipeline into a validated inference
service. Training remains in the notebooks. Inference always loads already-fitted
objects and never calls `fit`.

### Architecture

```mermaid
flowchart LR
    Client --> API[FastAPI]
    API --> GE[Great Expectations]
    GE --> Registry[MLflow Model Registry]
    Registry --> MinIO[MinIO artifacts]
    API --> PostgreSQL[(PostgreSQL prediction logs)]
    Prometheus --> API
    DVC --> MinIO
```

Docker Compose starts PostgreSQL, MinIO, MLflow, model registration, the API,
and Prometheus. The registered `champion` model is loaded by the API from MLflow;
the Task 2 artifact is mounted only into the one-time registration container.

### Project structure

```text
├── .dvc/                 # DVC remote and cache configuration
├── .github/workflows/    # CI pipeline
├── app/                  # FastAPI routes and Pydantic schemas
├── config/               # Non-secret application configuration
├── data/                 # Generated validation report and local prediction DB
├── examples/             # Working prediction payload
├── great_expectations/   # Incoming-data expectation suite
├── models/               # Reference prediction distribution for drift checks
├── monitoring/           # Prometheus and alert documentation
├── requirements/         # Pinned runtime and development dependencies
├── src/                  # Features, validation, registry, inference and utilities
├── Tasks/Task-02/        # Training notebooks and selected fitted artifacts
├── tests/                # Unit, data, model and API integration tests
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
└── pyproject.toml
```

### Start the complete stack

Requirements: Git and Docker Desktop with Docker Compose.

```bash
git clone <repository-url>
cd MLOps-Qafza-2026
docker compose up --build
```

Default local development credentials are defined through Compose defaults.
For non-local use, copy `.env.example` to `.env` and replace every password.
Do not commit `.env`.

After startup:

- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- Model info: <http://localhost:8000/model-info>
- Prometheus metrics: <http://localhost:8000/metrics>
- MLflow: <http://localhost:5000>
- MinIO console: <http://localhost:9001>
- Prometheus: <http://localhost:9090>

Stop the stack without deleting its data:

```bash
docker compose down
```

### Local Python setup

Python 3.11 is used by CI and Docker.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/requirements-dev.txt
```

Run the local artifact mode used by tests:

```bash
uvicorn app.main:app --reload
python -m src.cli predict --input examples/sample_order.json
```

Production containers set `MLOPS_MODEL_SOURCE=mlflow`; local artifact mode exists
only for offline development, testing, and initial registration.

### API example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/sample_order.json
```

The response includes a label, binary prediction, probability, request ID, and
the exact registered model version:

```json
{
  "request_id": "generated-uuid",
  "prediction": "on_time",
  "is_late": 0,
  "probability": 0.42,
  "model_version": "1"
}
```

Batch prediction accepts a JSON list at `POST /predict-batch`, with a maximum
size controlled by `api.max_batch_size`.

### Configuration

`config/config.yaml` is the source for paths and non-secret parameters. These
deployment values can be overridden by environment variables:

| Environment variable | Purpose |
|---|---|
| `MLOPS_MODEL_SOURCE` | `local` for tests or `mlflow` for the stack |
| `MLFLOW_TRACKING_URI` | MLflow tracking and registry endpoint |
| `DATABASE_URL` | SQLite or PostgreSQL prediction-log connection |
| `MLOPS_MODEL_NAME` | Registered model name |
| `MLOPS_MODEL_ALIAS` | Registry alias loaded by the service |
| `MLOPS_MODEL_THRESHOLD` | Fallback decision threshold |
| `MLOPS_MAX_BATCH_SIZE` | Maximum records per batch request |
| `MLOPS_LOG_LEVEL` | Python log level |

Secrets and connection strings belong in environment variables, never in Git.

### Validation and failure behavior

Pydantic checks the HTTP contract. Great Expectations then validates the exact
model-ready columns, allowed Brazilian state codes and payment types, numeric
ranges, and required values. Invalid requests are rejected with HTTP 422 before
the model is called.

To demonstrate failure handling, change `customer_state` in the sample request
to `XX`; the API returns 422. To demonstrate CI protection, temporarily change a
test assertion and run pytest; the command returns a non-zero exit code.

### DVC data versioning

The DVC pipeline packages the selected model, feature list, and metadata into a
versioned artifact bundle. It also validates the final training feature table
and writes a quality report tied to the exact data and code hashes:

```bash
dvc repro
dvc metrics show
```

The configured remote is the Compose MinIO `dvc` bucket. Set
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, then use `dvc push` or
`dvc pull`. `dvc.lock` ties the report to the exact input data and validation
code hashes.

### MLflow tracking and registry

`src/register_model.py` logs Task 2 parameters, test metrics, metadata, the
feature list, model signature, example input, and fitted pipeline. It registers
a model version and assigns the `champion` alias. Compose runs this step before
starting the API. For a manually started MLflow server:

```bash
python -m src.register_model
```

### Tests and quality checks

```bash
black --check app src tests
flake8 app src tests --max-line-length=88 --extend-ignore=E203,W503
pytest --cov=app --cov=src --cov-report=term-missing
pre-commit run --all-files
```

Tests cover feature creation, Great Expectations failures, data schema/ranges/
nulls/leakage, model loading and shape, exact notebook-artifact parity, CLI,
prediction persistence, API routes, batch behavior, bad payloads, and metrics.

### Logging, persistence, and monitoring

Logs are JSON-formatted to the console and a rotating file. Every prediction
records its sanitized input, output, probability, latency, request ID, and model
version. PostgreSQL stores the same record with an empty `actual_is_late` field
so the real outcome can be joined later.

Prometheus exposes request count, status, latency, prediction distribution,
probability histogram, and probability PSI. The PSI baseline comes from the
Task 2 held-out set. Alert thresholds and operational responses are documented
in `monitoring/ALERTS.md`.

### CI/CD

GitHub Actions runs formatting, linting, tests with coverage, and a Docker image
build on every push and pull request. On pushes to `main`, it also pushes the
image when `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are configured.
