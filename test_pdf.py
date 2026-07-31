import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

pdf_dir = Path("./artifacts/epstein_files_pdfs")
for pdf_path in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    pages = convert_from_path(pdf_path, first_page=1, last_page=2) # Test only first 2 pages
    
    doc_text = ""
    for i, page in enumerate(pages):
        img = np.array(page)
        img = img[:, :, ::-1].copy() 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        redaction_count = 0
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            if len(approx) >= 4 and (w > 50 and h > 15) and (w * h > 1000):
                redaction_count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.putText(img, "[REDACTED]", (x + 5, y + int(h/2) + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        text = pytesseract.image_to_string(img)
        doc_text += f"\n\n--- Page {i+1} ---\n\n" + text
        print(f"Page {i+1} processed with {redaction_count} redactions found.")
    
    print("Extracted Text Preview:")
    print(doc_text[:500])
