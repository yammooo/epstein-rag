1     ## Phase 3: Redacted Document Ingestion Pipeline

    2     

    3     This document outlines the strategy for implementing the requirements from pipeline_integration.md into the existing PART2.ipynb notebook.

    4     

    5     ## Goal Description



    6     The objective is to read PDF documents from ./artifacts/epstein_files_pdfs, extract images, identify redacted regions, explicitly tag them with [REDACTED], run OCR to get

          text, and finally chunk, embed, and inject these documents into the existing FAISS RAG index. Note that redactions are white boxes with black borders, so we will use     

          Canny Edge Detection to find them.

    7     

    8     ## User Review Required



    9     │ [!IMPORTANT]

   10     │ The OCR choice: The requirements mention olmocr (or an open-weight VLM). Since running a VLM like olmocr might require heavy GPU usage or API keys, I propose using     

          │ pytesseract (Tesseract OCR) as a lightweight, standard open-source OCR tool that easily integrates with python to extract text. If you prefer olmocr or another specific

          │ library, please let me know.

   11     

   12     │ [!WARNING]

   13     │ Tesseract requires a system-level dependency. We will need to run !sudo apt-get install -y tesseract-ocr poppler-utils (or equivalent) in the notebook environment.     

   14     

   15     │ [!NOTE]

   16     │ Saving Index: Currently the notebook stores chunks_df.parquet and embeddings.npy, but the FAISS index is built in-memory. I will update the code to append the new PDF  

          │ chunks to chunks_df and their embeddings to embeddings, saving both back to disk. Then we re-initialize the FAISS index.

   17     

   18     ## Proposed Changes

   19     

          ### PART2.ipynb



   21     We will add new cells at the end of the notebook (or right before the evaluation section) to perform the pipeline.

   22     

   23     #### [MODIFY] PART2.ipynb



   24     Cell 1: Dependencies

   25     

   26       # Install required libraries

   27       !pip install opencv-python pdf2image pytesseract langchain langchain-community

   28       # If on Colab / Linux, you may need:

   29       !sudo apt-get install -y tesseract-ocr poppler-utils

   31     

   32     Cell 2: PDF processing & OpenCV burn-in



>  33     

   34       import cv2

   35       import numpy as np

   36       import pytesseract

   37       from pdf2image import convert_from_path

   38       from pathlib import Path

   39     

   40       pdf_dir = Path("./artifacts/epstein_files_pdfs")

   41       parsed_texts = []

   42     

   43       for pdf_path in pdf_dir.glob("*.pdf"):

   44           print(f"Processing {pdf_path.name}...")

   45           pages = convert_from_path(pdf_path)

   46           

   47           doc_text = ""

   48           for i, page in enumerate(pages):

   49               # Convert PIL image to cv2 array

   50               img = np.array(page)

   51               # Convert RGB to BGR 

   52               img = img[:, :, ::-1].copy() 

   53               

   54               # Convert to grayscale

   55               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   56     

   57               # Apply Gaussian Blur to reduce noise before edge detection

   58               blurred = cv2.GaussianBlur(gray, (5, 5), 0)

   59     

   60               # Use Canny edge detection to find the black borders of the white boxes

   61               edges = cv2.Canny(blurred, 50, 150)

   62     

   63               # Find contours from the detected edges

   64               contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   65               

   66               for contour in contours:

   67                   # Approximate the contour to a polygon to check for rectangular shapes

   68                   epsilon = 0.02 * cv2.arcLength(contour, True)

   69                   approx = cv2.approxPolyDP(contour, epsilon, True)

   70                   

   71                   # Get bounding box

   72                   x, y, w, h = cv2.boundingRect(contour)

>  73                   

   74                   # Filter for large rectangular shapes (typical for redaction boxes)

   75                   # Checking if it has ~4 vertices helps ignore random squiggles or thick text

   76                   if len(approx) >= 4 and (w > 50 and h > 15) and (w * h > 1000):

   77                       # Draw a solid white box over the entire bounded area to ensure it's clean

   78                       cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

   79       

   80                       # Draw [REDACTED] text

   81                       cv2.putText(img, "[REDACTED]", (x + 5, y + int(h/2) + 5), 

   82                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

   83               

   84               # Run OCR on the modified image

   85               text = pytesseract.image_to_string(img)

   86               doc_text += f"\n\n--- Page {i+1} ---\n\n" + text

   87               

   88           parsed_texts.append({"doc_id": pdf_path.stem, "text": doc_text})

   90     

   91     Cell 3: RAG Integration

   92     

   93       import pandas as pd

   94       from langchain_text_splitters import RecursiveCharacterTextSplitter

   95     

   96       # Use existing tokenizer

   97       local_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(

   98           tokenizer=tokenizer,

   99           chunk_size=512,

  100           chunk_overlap=64,

  101           separators=["\n\n", "\n", ". ", " ", ""]

  102       )

  103     

  104       new_chunks = []

  105       for doc in parsed_texts:

  106           header = f"Source Document: {doc['doc_id']}\nContent: PDF File Extraction"

  107           chunks = local_splitter.split_text(doc["text"])

  108           

  109           for idx, c in enumerate(chunks):

  110               new_chunks.append({

  111                   "chunk_id": f"pdf_{doc['doc_id']}_{idx}",

  112                   "text_for_embedding": f"{header}\n\nChunk {idx+1}:\n{c}",

113                   "text_for_prompt": f"{header}\n\nChunk {idx+1}:\n{c}",

  114                   "subject": "PDF Document",

  115                   "from_name": "PDF",

  116                   "from_email": "PDF",

  117                   "to": "PDF",

  118                   "date": "Unknown",

  119                   "doc_ids": [doc["doc_id"]]

  120               })

  121     

  122       new_chunks_df = pd.DataFrame(new_chunks)

  123     

  124       # Compute embeddings for new chunks

  125       new_embeddings = compute_embeddings(new_chunks_df, embedding_model)

  126     

  127       # Append to existing data

  128       chunks_df = pd.concat([chunks_df, new_chunks_df], ignore_index=True)

  129       embeddings = np.vstack([embeddings, new_embeddings])

  130     

  131       # Save updated data to disk

  132       chunks_df.to_parquet(CHUNKS_DF_PATH, index=False)

  133       np.save(EMBEDDINGS_PATH, embeddings)

  134     

  135       # Rebuild FAISS index

  136       embedding_matrix = embeddings.astype("float32")

  137       index = faiss.IndexFlatIP(embedding_matrix.shape[1])

  138       index.add(embedding_matrix)

  139     

  140       print(f"Added {len(new_chunks_df)} chunks from PDF documents to the index.")

  142     

  143     ## Verification Plan

  144     

  145     ### Automated Tests

        

  146     Run the entire modified PART2.ipynb from start to finish. I will execute the notebook using jupyter nbconvert --to notebook --execute (or programmatically) and verify    

        that it completes without errors, producing the expected chunks_df.parquet and embeddings.npy updates.

  147     

  148     ### Manual Verification

        

> 149     1. Open the updated notebook and observe the new section.

113                   "text_for_prompt": f"{header}\n\nChunk {idx+1}:\n{c}",

  114                   "subject": "PDF Document",

  115                   "from_name": "PDF",

  116                   "from_email": "PDF",

  117                   "to": "PDF",

  118                   "date": "Unknown",

  119                   "doc_ids": [doc["doc_id"]]

  120               })

  121     

  122       new_chunks_df = pd.DataFrame(new_chunks)

  123     

  124       # Compute embeddings for new chunks

  125       new_embeddings = compute_embeddings(new_chunks_df, embedding_model)

  126     

  127       # Append to existing data

  128       chunks_df = pd.concat([chunks_df, new_chunks_df], ignore_index=True)

  129       embeddings = np.vstack([embeddings, new_embeddings])

  130     

  131       # Save updated data to disk

  132       chunks_df.to_parquet(CHUNKS_DF_PATH, index=False)

  133       np.save(EMBEDDINGS_PATH, embeddings)

  134     

  135       # Rebuild FAISS index

  136       embedding_matrix = embeddings.astype("float32")

  137       index = faiss.IndexFlatIP(embedding_matrix.shape[1])

  138       index.add(embedding_matrix)

  139     

  140       print(f"Added {len(new_chunks_df)} chunks from PDF documents to the index.")

  142     

  143     ## Verification Plan

  144     

  145     ### Automated Tests

        

  146     Run the entire modified PART2.ipynb from start to finish. I will execute the notebook using jupyter nbconvert --to notebook --execute (or programmatically) and verify    

        that it completes without errors, producing the expected chunks_df.parquet and embeddings.npy updates.

  147     

  148     ### Manual Verification

        

> 149     1. Open the updated notebook and observe the new section.

150     2. Check if a query about redacted PDFs returns relevant answers containing the [REDACTED] tag. 

