---
title: Resume Analyser
emoji: 🚀
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
nned: false
---

 out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## GitHub Actions: deploy to GCP Cloud Run (extra credit)

The workflow `.github/workflows/deploy-cloud-run.yml` builds the Docker image, pushes it to Artifact Registry, and deploys to Cloud Run on pushes to `main` or on manual runs.

Setup steps:
- Enable the Cloud Run Admin API and Artifact Registry API in your project.
- Create (or let the workflow create) a Docker Artifact Registry repository named `resume-comparator` in your `$GCP_REGION`.
- Create a service account with roles `roles/run.admin`, `roles/iam.serviceAccountUser`, and `roles/artifactregistry.admin`. Generate a JSON key for this service account.
- Add GitHub secrets: `GCP_PROJECT_ID`, `GCP_REGION`, `CLOUD_RUN_SERVICE` (desired service name), and `GCP_SA_KEY` (the JSON key contents).

Usage:
- Push to `main` or run the workflow manually. On success, the job prints the Cloud Run service URL you can share
