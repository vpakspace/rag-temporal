#!/bin/bash
# Launch Streamlit UI for RAG + Temporal unified system
cd "$(dirname "$0")"
streamlit run streamlit_app.py --server.port 8503
