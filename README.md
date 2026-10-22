# AI-Assisted Medical Imaging and Diagnosis System

An AI/ML capstone project that explores medical image analysis, deep learning-based classification, explainable AI, DICOM image processing, and human-in-the-loop review workflows.

> **Educational use only — not for clinical use.** This project is a research and educational prototype. It is not intended to diagnose patients, guide treatment, or replace qualified medical professionals.

## 1. Project Overview

The AI-Assisted Medical Imaging and Diagnosis System is designed to demonstrate how deep learning can assist with the analysis of medical images such as chest X-rays.

The application is organized into a React frontend and a Python FastAPI backend. The backend is responsible for image validation, preprocessing, model inference, explainability, DICOM processing, and review-related operations.

The frontend provides an interface for uploading images, viewing analysis results, inspecting explanation heatmaps, and recording review decisions.

The system is designed around a human-in-the-loop workflow: AI-generated results are presented for review rather than treated as definitive medical conclusions.

## 2. Project Objectives

- Develop a web application for medical image upload and visualization.
- Integrate a suitable pretrained deep learning model for image classification.
- Apply image preprocessing before inference.
- Explore Grad-CAM for visualizing regions that influence model predictions.
- Support DICOM image parsing and conversion into a format suitable for analysis.
- Provide a review workflow with approval/rejection actions.
- Maintain an audit trail of review decisions.
- Apply software engineering practices such as API validation, testing, environment configuration, and containerization.
- Document model limitations, dataset considerations, potential bias, and intended use.

## 3. Key Features

The following describes the project's intended capabilities. Each feature should be marked complete only after it has been implemented and tested.

### Medical Image Upload

- Upload supported image files through the frontend.
- Validate file formats and reject invalid or corrupted files.
- Display the uploaded image before analysis.
- Provide clear loading and error states.

### AI-Based Image Classification

- Preprocess the uploaded image according to the selected model's requirements.
- Run inference using a compatible pretrained medical imaging model.
- Return the model's output label and associated score.
- Identify the model and document its source and intended use.

The output is model-generated information, not a confirmed diagnosis. Scores must not be interpreted as calibrated clinical probabilities unless calibration has been independently evaluated.

### Explainable AI with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) can be used to generate a heatmap showing image regions that contribute to a selected model output.

The intended workflow is:

1. Run inference on the input image.
2. Select the target class and an appropriate feature layer.
3. Calculate the Grad-CAM activation map.
4. Resize the map to match the displayed image.
5. Overlay the heatmap on the original image.

The visualization helps inspect model behavior. It does not prove that a prediction is correct or that the highlighted region represents a medically meaningful finding.

### DICOM Image Processing

DICOM (Digital Imaging and Communications in Medicine) is a standard for medical imaging data and associated metadata.

The backend can use `pydicom` to read supported DICOM files, access pixel data, and prepare image arrays for visualization or inference.

DICOM handling must account for pixel representation, grayscale interpretation, rescaling, and compression. Not every DICOM file can be decoded without additional libraries.

### Doctor Review Workflow

The review interface is intended to support:

- Reviewing the uploaded scan and model output.
- Recording an approval or rejection decision.
- Associating the decision with a study identifier.
- Recording a timestamp and optional reviewer notes.
- Retrieving previous review records.

An approval or rejection records a workflow decision; it does not establish a clinical diagnosis.

### Audit Logging

Review records should capture the relevant study identifier, model output, reviewer decision, timestamp, and any supported notes.

The database implementation should be tested to ensure records persist across application restarts. Production systems would require stronger access controls, data protection, and tamper-resistant audit logging.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Frontend | React |
| Frontend tooling | Vite |
| Frontend language | JavaScript |
| Backend | Python |
| API framework | FastAPI |
| API server | Uvicorn |
| Request validation | Pydantic |
| Image processing | Pillow, NumPy |
| Medical image parsing | pydicom |
| Deep learning | PyTorch and Torchvision, if used by the selected model |
| Explainability | Grad-CAM implementation |
| Database | SQLite, if configured |
| Testing | pytest and HTTPX |
| Version control | Git and GitHub |
| Containerization | Docker, if configured |

## 5. System Architecture

The application follows a frontend-backend architecture.

```text
User
  |
  v
React + Vite Frontend
  |
  | HTTP requests / multipart file upload
  v
FastAPI Backend
  |
  +--> Input validation
  |
  +--> Image preprocessing
  |
  +--> Model inference
  |       |
  |       +--> Prediction and score
  |       +--> Grad-CAM explanation
  |
  +--> DICOM processing
  |
  +--> Review workflow
  |
  +--> SQLite database
          |
          +--> Review and audit records
```

The frontend is responsible for presenting information and collecting user interactions. The backend performs processing and inference, and the database stores supported persistent review records.

## 6. Project Structure

```text
AI_Medical_Imaging/
|
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   |-- main.py
|   |   |-- config.py
|   |   |-- database.py
|   |   |-- schemas.py
|   |   |
|   |   |-- api/
|   |   |   |-- __init__.py
|   |   |   |-- health.py
|   |   |   |-- analysis.py
|   |   |   |-- reviews.py
|   |   |   |-- dicom.py
|   |   |
|   |   |-- services/
|   |   |   |-- __init__.py
|   |   |   |-- analysis_service.py
|   |   |   |-- review_service.py
|   |   |   |-- dicom_service.py
|   |   |
|   |   |-- ml/
|   |   |   |-- __init__.py
|   |   |   |-- model_loader.py
|   |   |   |-- inference.py
|   |   |   |-- gradcam.py
|   |   |
|   |   |-- utils/
|   |       |-- __init__.py
|   |       |-- image_processing.py
|   |       |-- validators.py
|   |
|   |-- tests/
|   |-- models/
|   |-- data/
|   |-- requirements.txt
|   |-- .env.example
|   |-- .gitignore
|   |-- Dockerfile
|
|-- frontend/
|   |-- public/
|   |-- src/
|   |   |-- components/
|   |   |-- services/
|   |   |-- App.jsx
|   |   |-- App.css
|   |   |-- main.jsx
|   |
|   |-- index.html
|   |-- package.json
|   |-- package-lock.json
|   |-- .env.example
|   |-- .gitignore
|
|-- docs/
|-- tests/
|-- .gitignore
|-- README.md
```

This is a reference structure. Adjust it to match the actual files in your repository; remove entries that do not exist.

## 7. Prerequisites

Install the following tools:

- Python 3.11 or another version supported by your chosen ML dependencies.
- Node.js and npm.
- Git.
- Visual Studio Code.
- Docker Desktop, if using the Docker configuration.

Check your installations:

```powershell
python --version
node --version
npm --version
git --version
```

Python machine-learning dependencies can have version-specific compatibility requirements. Install the version of PyTorch and other packages supported by the chosen model.

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_Medical_Imaging
```

Replace the placeholder with your actual repository URL.

### Step 2: Configure the Backend

Open a terminal in the backend directory:

```powershell
cd backend

python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install any additional model-specific dependencies according to the selected model's official instructions.

If the virtual environment already exists, activate it instead of recreating it.

### Step 3: Configure Environment Variables

If the backend uses environment variables, create a local `.env` file based on `.env.example`.

For example:

```dotenv
APP_ENV=development
DATABASE_URL=sqlite:///./medical_imaging.db
MODEL_PATH=./models/model_weights.pth
```

These names are examples. Keep only variables that the application actually reads, and ensure their values match the implementation.

Do not commit real credentials, secrets, private patient data, or restricted datasets.

### Step 4: Start the Backend

From the `backend` directory:

```powershell
uvicorn app.main:app --reload
```

The default local API address is:

```text
http://127.0.0.1:8000
```

If the application uses a different module path or port, update the command accordingly.

FastAPI interactive API documentation is normally available at:

```text
http://127.0.0.1:8000/docs
```

### Step 5: Configure the Frontend

Open a second terminal:

```powershell
cd frontend
npm install
```

Create a local `.env` file if the frontend uses environment variables:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The frontend must read this variable in its API client. Vite exposes variables prefixed with `VITE_` to client-side code, so never put secrets in them.

### Step 6: Start the Frontend

```powershell
npm run dev
```

Open the local URL printed by Vite, commonly:

```text
http://localhost:5173
```

Keep both the frontend and backend terminals running during local testing.

## 9. API Reference

The following endpoints describe the intended API contract. Verify the actual routes in FastAPI's `/docs` page and update this section to match the implemented endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check backend availability |
| POST | `/api/analyze` | Upload an image and request analysis |
| GET | `/api/reviews` | Retrieve review history |
| POST | `/api/reviews` | Record a review decision |
| POST | `/api/dicom/analyze` | Process a DICOM upload |

### Image Analysis

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@sample_xray.png"
```

The request field name must match the backend's implementation.

A successful response should communicate the actual model output, model identifier, study identifier, and explanation location where available.

Example response schema:

```json
{
  "study_id": "generated-study-id",
  "model_name": "selected-model",
  "prediction": "actual-model-output",
  "confidence": 0.0,
  "heatmap_url": "/api/studies/generated-study-id/heatmap"
}
```

The example illustrates the response format only. The values are placeholders, not actual inference results. Use the response fields defined in `schemas.py`.

### Review Decision

A review request should include the relevant study identifier and a supported decision such as `approve` or `reject`.

The backend should validate the request, save the decision and timestamp, and return the stored record.

## 10. Dataset and Model

Document the dataset and model actually used in the implementation.

Potential research datasets include:

- **CheXpert:** chest radiographs with associated labels.
- **NIH ChestX-ray14:** chest radiographs with associated annotations.

Official resources:

- CheXpert: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-Nihcc

Before using a dataset, review its license, access terms, permitted uses, and attribution requirements.

The model documentation should identify:

- Model architecture and source.
- Pretrained weights and their license.
- Supported image dimensions and preprocessing.
- Output classes and interpretation.
- Training or fine-tuning performed, if any.
- Dataset provenance.
- Evaluation metrics and test methodology.
- Known limitations and potential biases.

Do not claim that a model was trained or fine-tuned on a dataset unless that work was actually performed.

## 11. Image Preprocessing and Explainability

Image preprocessing must match the model's expected input. Relevant operations may include:

1. Decoding the uploaded image.
2. Converting image channels when appropriate.
3. Resizing to the required dimensions.
4. Scaling pixel intensities.
5. Applying the correct normalization.
6. Converting the image into a model tensor.

Grad-CAM must be computed from the selected model and a compatible feature layer. The resulting heatmap should be aligned with the displayed image.

The model's output and heatmap should be reproducible for the same input under the same model and preprocessing configuration, subject to normal implementation and numerical considerations.

## 12. DICOM Processing

DICOM files may contain pixel data, image dimensions, modality information, and identifying metadata.

The implementation should:

- Validate that the file can be parsed.
- Handle unsupported transfer syntaxes and compressed pixel data appropriately.
- Apply required pixel-value transformations when appropriate.
- Handle grayscale presentation and image orientation correctly.
- Avoid exposing patient-identifying metadata unnecessarily.
- Return useful errors for unsupported files.

A DICOM parser is not the same as a complete DICOM server integration. If Orthanc is not implemented, describe the current capability as DICOM file parsing and document Orthanc integration as future work.

## 13. Database and Audit Logging

If SQLite is used, document the actual database schema and stored fields.

A review record may contain:

- Study identifier.
- Model output associated with the reviewed study.
- Review decision.
- Reviewer identifier, where supported.
- Optional notes.
- Timestamp.

The implementation should use parameterized database operations, validate incoming data, and handle database errors.

For real medical deployments, authentication, authorization, secure storage, retention controls, and stronger audit protections would be required.

## 14. Testing

Run backend tests from the backend directory:

```powershell
pytest -v
```

Run frontend build checks from the frontend directory:

```powershell
npm run build
```

If the frontend defines a test script, run it as well:

```powershell
npm test
```

Recommended test cases include:

- Backend health check.
- Successful upload of a supported image.
- Invalid file extension or MIME type.
- Corrupted image.
- Missing upload.
- Inference failure.
- Grad-CAM generation and output dimensions.
- Invalid review decisions.
- Persistence of review history.
- Valid and invalid DICOM files.
- Frontend loading, success, and error states.

Record actual results. Do not report tests as passing unless you ran them.

## 15. Docker

If Docker is implemented, build and run the backend using the project's actual Dockerfile and configuration.

Example:

```bash
docker build -t medical-imaging-backend ./backend
docker run --rm -p 8000:8000 medical-imaging-backend
```

The Docker image must include the required application dependencies and use the correct application startup command.

If a model file or database is stored outside the image, configure the necessary volume mounts or model-download process.

Do not include large datasets, model weights, or sensitive information in the image without a clear reason.

## 16. Limitations and Responsible Use

This project has important limitations:

- Model predictions depend on the training data, labels, preprocessing, and selected architecture.
- Performance may vary across hospitals, patient populations, acquisition devices, and imaging protocols.
- A model score does not automatically represent a calibrated probability.
- Grad-CAM is an explanation aid, not proof of correctness or causality.
- DICOM compatibility depends on the transfer syntax and image characteristics.
- A successful software test does not establish clinical validity.
- A review button does not replace a qualified clinician's assessment.

This prototype must not be used to make clinical decisions. Clinical use would require appropriate validation, regulatory and privacy review, security controls, and qualified professional oversight.

## 17. Future Improvements

Potential extensions include:

- Integration with Orthanc for DICOM study retrieval.
- A robust DICOMweb workflow.
- Fine-tuning and evaluating a medical image classifier.
- A 3D U-Net segmentation pipeline using MONAI.
- Dataset and model versioning using DVC.
- Automated model evaluation and bias analysis.
- Authentication and role-based access control.
- Improved audit security and monitoring.
- Container orchestration and reproducible deployment.
- Evaluation on an appropriately separated test dataset.

These are future improvements unless they are already implemented and tested.

## 18. Submission Deliverables

The project submission should contain:

- Source code for the backend and frontend.
- Dependency and setup instructions.
- Model source, license information, and configuration.
- Dataset source and usage instructions.
- Test code and actual test results.
- Screenshots of implemented functionality.
- Architecture and implementation documentation.
- Limitations and educational-use disclaimer.
- Docker configuration, if implemented.
- Demo recording, if available.

Large model weights and datasets should be linked or documented when redistribution is not permitted.

## 19. Author and Project Information

**Project:** AI-Assisted Medical Imaging and Diagnosis System  
**Project ID:** AIML-02  
**Track:** AI / Machine Learning  
**Author:** Sneha Poojary  
**Purpose:** Academic capstone and educational prototype

---

**Disclaimer:** This software is provided for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.
