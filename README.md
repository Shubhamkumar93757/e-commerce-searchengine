# e-commerce-searchengine

This is my new project


ecommerce-multimodal-search/
├── data/
│   ├── raw/                    # Raw product images + CSVs
│   ├── processed/              # Cleaned data
│   └── embeddings/             # Stored vectors (ONNX format)
├── models/
│   ├── vision_encoder.py       # ResNet50 wrapper
│   ├── text_encoder.py         # SBERT wrapper
│   ├── shared_space.py         # Projection layer
│   └── convert_to_onnx.py      # Conversion script
├── pipeline/
│   ├── offline_batch.py        # Batch embedding generation
│   ├── vector_db_loader.py     # Push to Qdrant/Milvus
│   └── training_loop.py        # Contrastive loss training
├── api/
│   ├── main.py                 # FastAPI server
│   ├── inference.py            # ONNX runtime wrapper
│   └── search.py               # ANN search endpoints
├── frontend/
│   ├── app.py                  # Streamlit UI (optional)
│   └── static/                 # React/Vue (for production)
├── docker-compose.yml          # Qdrant + API services
├── requirements.txt
├── .env.example
└── README.md