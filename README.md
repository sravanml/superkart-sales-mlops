---
title: SuperKart Sales Estimator
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SuperKart sales estimator

Estimate total sales for a product at one of four existing stores.

## Run locally
Use Python 3.11, install with `pip install -r requirements.txt`,
then run `python pipeline.py validate`, `python pipeline.py prepare`,
`python pipeline.py train`, and `streamlit run app.py` from the project root.

## Deploy
Push this project to the default branch `main`. Run **SuperKart training** in
GitHub Actions and verify that it commits the model. In Streamlit Community
Cloud, select the repository, branch `main`, and entry point `app.py`.

For Hugging Face Spaces, create a public Space using the Docker SDK, then push
this project including Dockerfile and the models/ folder created by training.
The README front matter sets the Docker app port to 7860.
