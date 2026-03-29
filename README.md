# AI Practitioner's Portfolio: Machine Learning & LLM Engineering

이 저장소는 **전통적인 ML/DL부터 최신 Agentic AI(LangGraph)와 MLOps 서빙**까지, AI 엔지니어링 전 분야의 핵심 기술을 실습하고 증명하는 공간입니다. 각 폴더는 특정 주제에 대한 깊은 이론 학습과 실제 동작하는 코드 구현물로 구성되어 있습니다.

---

## Repository Structure & Module Explorer

```bash
.
├── MachineLearning_DeepLearning_CV
│   ├── 01_Data_and_ML           # Pandas 기반 데이터 처리, Ensemble(RF, XGBoost, PCA)
│   ├── 02_DL_Fundamentals      # RNN, Seq2Seq, Transfer Learning, 최적화 구현
│   ├── 03_Computer_Vision_Adv  # ResNet, CAM/Grad-CAM, YOLOv1, Faster R-CNN, U-Net
│   └── 04_Generative_Models    # VAE, GAN, cGAN 등 데이터 생성 모델 학습
└── NLP_MLOps&Serving
    ├── 01_NLP_and_Transformers # Attention, Transformer, BERT/GPT 사전학습 (Pre-training)
    ├── 02_LLM_FineTuning      # LoRA/QLoRA (Gemma, BERT), PEFT 경량화 학습
    ├── 03_RAG_and_Agents      # LangChain, LangGraph 에이전트, Self-Corrective RAG
    └── 04_MLOps_and_Serving    # vLLM/Triton 서빙, FastAPI, Docker 인프라
```

---

## Detailed Technical Competencies

### 1️. Advanced Computer Vision & Generation
단순 이미지 분류를 넘어 시각 인지 및 객체 생성 전반을 다룹니다.
- **Explainable AI (XAI)**: `CAM`, `Grad-CAM`을 활용한 CNN의 피처 시각화 및 판단 근거 분석.
- **Object Detection & Segmentation**: `YOLOv1`, `Faster R-CNN`, `U-Net`, `Mask R-CNN` 등 핵심 감지 아키텍처 구현.
- **Generative AI 기초**: `VAE`와 `GAN(cGAN)`을 이용하여 잠재 공간(Latent Space)을 학습하고 가짜 이미지를 생성.

### 2️. NLP Foundations & LLM Engineering
텍스트를 이해하고 처리하는 Transformer 아키텍처의 밑바닥부터 대형 언어 모델의 효율적 학습을 다룹니다.
- **Transformers from scratch**: `Scaled Dot-Product Attention`, `Multi-Head Attention` 직접 구현.
- **Pre-training Specialist**: `BERT(ALBERT)`, `GPT`의 사전 학습 메커니즘 분석 및 커스텀 사전 학습 데이터셋 루프 구현.
- **Efficient Fine-tuning (PEFT)**: `LoRA` 및 `QLoRA` 기법으로 `Gemma` 등 대형 모델을 효율적으로 미세 조정.

### 3️. Agentic AI & RAG (Orchestration)
단발성 답변을 넘어, 스스로 추론하고 교정하는 고도화된 워크플로우를 설계합니다.
- **LangGraph Application**: `Self-Corrective RAG`, `ReAct Agent` 개발을 통한 스테이트풀(Stateful) 에이전트 워크플로우 제어.
- **Search Optimization**: `Hybrid Search`, `Multi-vector Search`, `Re-ranking`을 통한 고품질 검색 증강 시스템(RAG) 구축.
- **Trace & Monitoring**: `LangSmith` 연동을 통한 성능 트래킹 및 트러블슈팅 경험.

### 4️. MLOps & High-Performance Inference
모델을 서비스 환경에 배포하고 확장 가능한 인프라를 구축하는 능력을 갖추었습니다.
- **High-Throughput Serving**: `vLLM`, `NVIDIA Triton Inference Server` 기반의 고성능 모델 서빙.
- **FastAPI Model Serving**: 확장성을 고려한 API 설계 및 비동기 처리 구현.
- **Deployment Infrastructure**: `Docker` 컨테이너화 및 `Streamlit` 기반의 AI 데모 프로토타이핑.

---

## Applied Tech Stack

- **Model Frameworks**: PyTorch, HuggingFace Transformers, PEFT, BitsAndBytes
- **AI Orchestration**: LangChain, LangGraph, LangSmith
- **Inference & Ops**: vLLM, NVIDIA Triton, Docker, FastAPI
- **Data Engineering**: Pandas, Scikit-learn, VectorDB (Pinecone, FAISS 등)

---

## Author
**[이승완]**
- AI Engineer
- Contact: [wandorigas93@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/%EC%9D%B4%EC%8A%B9%EC%99%84-seungwan-lee-2016b9383/]
- Blog: [https://velog.io/@wandorigas93]
