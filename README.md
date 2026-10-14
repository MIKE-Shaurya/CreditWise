# CreditWise — Loan Approval Model

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io/)
[![status](https://img.shields.io/badge/status-active-brightgreen)](#)

A binary classifier that predicts whether a loan application will be approved, based on applicant financials, credit history, and loan details. The trained model is served through a FastAPI backend and a Streamlit web app.

## 🚀 Live demo

| | Link |
| --- | --- |
| **Web app (Streamlit)** | https://creditwise-1.streamlit.app/ |
| **API (FastAPI on Render)** | https://creditwise-1.onrender.com |
| **API docs (Swagger)** | https://creditwise-1.onrender.com/docs |

> **Note:** the API runs on Render's free tier, which sleeps after a period of inactivity. The first prediction after a break can take 30–60 seconds while the service wakes up.

## Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running locally](#running-locally)
- [API reference](#api-reference)
- [Dataset columns](#dataset-columns)
- [Model performance](#model-performance)
- [Making a prediction in Python](#making-a-prediction-in-python)
- [Models](#models)
- [Deployment](#deployment)

## Architecture

```
┌──────────────────┐   HTTP (JSON)   ┌──────────────────┐      ┌──────────────┐
│  Streamlit UI    │ ──────────────▶ │  FastAPI service │ ───▶ │ models/      │
│  (Streamlit      │ ◀────────────── │  (Render)        │      │ trained model│
│   Community      │   prediction    │  /predict        │      │ + scaler     │
│   Cloud)         │                 └──────────────────┘      └──────────────┘
└──────────────────┘
```

The UI collects applicant details, sends them to the API, and displays whether the application is likely to be approved along with the approval probability.

## Project structure

```
CreditWise/
├── README.md
├── render.yaml               # Render deployment config for the API
├── requirements.txt          # full environment (notebooks + training)
├── requirements-api.txt      # API-only dependencies
├── requirements-ui.txt       # Streamlit UI-only dependencies
├── api/                      # FastAPI service (serves /predict, /options, ...)
├── app/                      # Streamlit web app
├── models/
