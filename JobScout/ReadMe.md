# JobScout — Automated Job Scoring and Digest Pipeline

An independent learning project exploring agentic AI, LangGraph, and 
automated pipeline design — built to develop hands-on skills in 
multi-step LLM workflows and task orchestration.

---

## What This Project Does

JobScout is an agentic pipeline that automatically discovers job postings, 
scores them against a resume, and delivers a daily digest with fit scores 
and keyword analysis — removing the manual work of sifting through job boards.

---

## How It Works
```
┌─────────────────────────────────────────────────────┐
│                   JobScout Agent                    │
│                  (LangGraph Pipeline)               │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐
│   Scraper Node  │  Discovers job postings from target sources
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Resume Reader   │  Parses and extracts resume content
│     Node        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Scorer Node    │  Scores each job against resume using LLM
│                 │  (fit score + keyword gap analysis)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reporter Node   │  Formats results into a structured digest
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Email Sender    │  Delivers daily digest to inbox
│     Node        │
└─────────────────┘
```

---

## Planned Tools Used

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM Backend | Ollama (local) |
| Resume Parsing | PyMuPDF |
| Job Scraping | Serper API |
| API Layer | FastAPI |
| Email Delivery | smtplib / Gmail |

---

## Status

Work in progress. This README will be updated as each node is built and tested.

---
