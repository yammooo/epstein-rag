## Phase 3: Redacted Document Ingestion Pipeline

**Objective:** Process the raw PDFs from `./artifacts/epstein_files_pdfs`, explicitly tag redacted regions with `[REDACTED]`, extract structured text, and ingest the parsed documents into the existing RAG vector space.

### 1. Requirements & Dependencies
Ensure the following libraries are installed in your local environment:
*   **Vision & Image Processing:** `opencv-python`, `numpy`, `pdf2image` (or `PyMuPDF` for fast rendering)
*   **OCR / Text Extraction:** `olmocr` (or your chosen open-weight VLM)
*   **RAG Architecture:** `langchain`, `langchain-community`, `faiss-cpu` (or `faiss-gpu`), and your existing open-source embedding model.

### 2. Pipeline Execution Steps

#### Step 1: PDF to Image Extraction
*   Target directory: `./artifacts/epstein_files_pdfs/`
*   Iterate through each `.pdf` file.
*   Convert each page into a high-resolution image array to prepare it for computer vision processing.

#### Step 2: Redaction "Burn-In" (OpenCV)
*   **Thresholding:** Convert the page image to grayscale and apply a binary threshold to isolate dark regions.
*   **Contour Detection:** Use `cv2.findContours()` to identify large, solid black rectangular blocks. Set a minimum area threshold to ensure you are catching redactions and not thick fonts or separator lines.
*   **Modification:** 
    *   For each detected redaction bounding box, draw a solid white rectangle (`cv2.rectangle`) to overwrite the black ink.
    *   Draw the text `[REDACTED]` (`cv2.putText`) inside the newly created white space, using a standard font scale.

#### Step 3: OCR Parsing
*   Pass the modified (burned-in) images through the OCR model pipeline.
*   Because the image now contains clean text reading `[REDACTED]`, the OCR model will seamlessly parse it as part of the natural reading order without hallucinating characters or breaking the layout.
*   Save the output as clean Markdown strings.

#### Step 4: RAG Vector Space Integration
*   **Chunking:** Pass the generated Markdown strings into a LangChain text splitter (e.g., `MarkdownTextSplitter` or `RecursiveCharacterTextSplitter`).
*   **Embedding:** Generate vector embeddings for the new document chunks using your local open-source embedder.
*   **Indexing:** Append the new vectors to your existing FAISS index. 
*   Save the updated index to disk to ensure the new files are immediately available for the retrieval pipeline.

---
*Note: This cell outlines the architecture. Execute the subsequent code blocks to run the individual OpenCV transformations and LangChain integrations.*