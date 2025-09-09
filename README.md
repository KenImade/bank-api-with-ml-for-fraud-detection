# Banking API with FastAPI + AI Fraud Detection

This project is the result of a course I completed on building a production-ready banking API with **FastAPI**, combined with **AI-powered fraud detection**.  

What I loved about this course is that it wasn’t just about building endpoints—it was about learning how to **design, secure, and scale a complete banking system** that feels close to something you’d find in the real world.  

---

## 📌 What I Learned

Throughout the project, I gained hands-on experience with:

- **API Architecture & Design**
  - How to structure a robust banking API using **domain-driven design**
  - Organizing code with scalability and maintainability in mind  

- **Security**
  - Implementing **JWT authentication** and **OTP verification**
  - Adding **rate limiting** to protect against brute-force and abuse  

- **Banking Features**
  - Account creation, deposits, withdrawals, transfers, and statements
  - Virtual card management (create, activate, block, top-up)
  - User profile management, Next of Kin info, and **KYC**  

- **Machine Learning**
  - Building a pipeline to analyze transactions in real time
  - Detecting fraudulent transactions with **scikit-learn**
  - Managing the full ML lifecycle with **MLflow**  

- **Deployment & Scaling**
  - Containerizing everything with **Docker**
  - Managing traffic with **Traefik** (reverse proxy + load balancing)
  - Using **Celery + RabbitMQ/Redis** for background tasks like:
    - Sending email notifications
    - Generating PDFs
    - Running ML training in the background  

- **Monitoring & Logging**
  - Logging best practices with **Loguru**
  - Keeping track of the system in production  

---

## ⚡ Key Features

- **Core Banking**
  - Create accounts, deposits, withdrawals, transfers, and statements  

- **Virtual Cards**
  - Card creation, activation, blocking, and top-ups  

- **Fraud Detection (AI/ML)**
  - Real-time transaction risk analysis using ML models  

- **Background Workers**
  - Email notifications
  - PDF generation
  - Fraud model training in the background  

- **Deployment Ready**
  - Docker Compose setup with Traefik
  - Scalable worker architecture  

- **ML Ops**
  - Train, evaluate, and deploy models with MLflow  

---

## 🛠️ Tech Stack

- **FastAPI** & **SQLModel** – API development + database modeling  
- **PostgreSQL** + **Alembic** – persistent storage + migrations  
- **Docker** + **Traefik** – containerization & reverse proxy  
- **Celery** + **RabbitMQ/Redis** – distributed task queues  
- **Scikit-learn** – fraud detection models  
- **MLflow** – experiment tracking & model deployment  
- **Pydantic V2** – data validation  
- **JWT** & **OTP** – authentication flows  
- **Cloudinary** – media storage  
- **Loguru** – logging  

---

## 🚀 Why This Project Matters

Most tutorials cover only the basics of FastAPI, but this course/project really helped me understand how to **connect the dots**:  

- From **API development → system design → ML integration → production deployment**.  
- It’s not just a toy app—it’s structured to look like something that could scale in the real world.  

---

## ✅ Outcome

By the end, I built a **secure, scalable banking API** with **AI-powered fraud detection** that I can proudly showcase in my portfolio.  

This project made me more confident about:  

- Designing real-world APIs  
- Deploying with containers  
- Integrating ML into production systems  

No more “basic tutorials”—this was about **building something real**.  
