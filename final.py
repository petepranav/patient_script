import csv
import json
import base64
import io
import re
import os
import random
import string
from datetime import datetime, timedelta
import requests
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv

import cv2
import numpy as np
import fitz
from PIL import Image
from ultralytics import YOLO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pytesseract
import threading
import tempfile

# === SELENIUM IMPORTS FOR NPI EXTRACTION ===
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# === UTILITY FUNCTIONS ===
def safe_strip(value, default=""):
    """
    Safely strip a value, handling None and non-string types
    Returns the stripped string or default value if input is None/invalid
    """
    try:
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return default

# === ENHANCED LOGGING SYSTEM ===
class Logger:
    """Enhanced logging system with colors and better formatting"""
    
    # ANSI color codes
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'RESET': '\033[0m'
    }
    
    def __init__(self):
        self.log_file = None
        self.setup_log_file()
    
    def setup_log_file(self):
        """Setup log file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, f"processing_log_{timestamp}.txt"), "w", encoding='utf-8')
    
    def _write_to_file(self, message):
        """Write message to log file"""
        if self.log_file:
            # Remove ANSI color codes for file output
            clean_message = re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', message)
            self.log_file.write(clean_message + "\n")
            self.log_file.flush()
    
    def _colorize(self, text, color):
        """Apply color to text"""
        return f"{Logger.COLORS.get(color, '')}{text}{Logger.COLORS['RESET']}"
    
    def _get_timestamp(self):
        """Get formatted timestamp"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _print_separator(self, char="=", length=80):
        """Print a separator line"""
        message = char * length
        print(self._colorize(message, 'CYAN'))
        self._write_to_file(message)
    
    def header(self, title):
        """Print a header with title"""
        self._print_separator()
        centered_title = f" {title} ".center(80, " ")
        message = f"║{centered_title}║"
        print(self._colorize(message, 'BOLD'))
        self._write_to_file(message)
        self._print_separator()
    
    def info(self, message, doc_id=None):
        """Print info message"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'BLUE')} ℹ️  {message}"
        print(message)
        self._write_to_file(message)
    
    def success(self, message, doc_id=None):
        """Print success message"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'GREEN')} ✅ {self._colorize(message, 'GREEN')}"
        print(message)
        self._write_to_file(message)
    
    def error(self, message, doc_id=None):
        """Print error message"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'RED')} ❌ {self._colorize(message, 'RED')}"
        print(message)
        self._write_to_file(message)
    
    def warning(self, message, doc_id=None):
        """Print warning message"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'YELLOW')} ⚠️  {self._colorize(message, 'YELLOW')}"
        print(message)
        self._write_to_file(message)
    
    def progress(self, message, doc_id=None):
        """Print progress message"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'PURPLE')} 🔄 {message}"
        print(message)
        self._write_to_file(message)
    
    def data(self, title, data, doc_id=None):
        """Print formatted data"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'CYAN')} 📊 {self._colorize(title, 'BOLD')}"
        print(message)
        self._write_to_file(message)
        if isinstance(data, dict):
            for key, value in data.items():
                message = f"     └─ {self._colorize(key, 'CYAN')}: {value}"
                print(message)
                self._write_to_file(message)
        else:
            message = f"     └─ {data}"
            print(message)
            self._write_to_file(message)
    
    def step(self, step_num, total_steps, description, doc_id=None):
        """Print step progress"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        progress_bar = f"[{step_num}/{total_steps}]"
        message = f"{self._colorize(prefix, 'PURPLE')} 📋 {self._colorize(progress_bar, 'BOLD')} {description}"
        print(message)
        self._write_to_file(message)
    
    def fax_status(self, is_faxed, fax_source=None, doc_type=None, doc_id=None):
        """Print fax status with special formatting"""
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        
        if is_faxed:
            icon = "📠"
            status_text = self._colorize("FAXED DOCUMENT", 'YELLOW')
            details = f"Source: {fax_source or 'Unknown'}"
        else:
            icon = "📄"
            status_text = self._colorize("NON-FAXED DOCUMENT", 'BLUE')
            details = f"Type: {doc_type or 'Unknown'}"
        
        message = f"{self._colorize(prefix, 'CYAN')} {icon} {status_text} | {details}"
        print(message)
        self._write_to_file(message)
    
    def close(self):
        """Close the log file"""
        if self.log_file:
            self.log_file.close()

# Create a global logger instance
logger = Logger()

def configure_physician_group(pg_id, pg_name, pg_npi):
    """
    Configure physician group parameters for processing
    
    Args:
        pg_id (str): Physician Group Company ID
        pg_name (str): Physician Group Name  
        pg_npi (str): Physician Group NPI number
    """
    global PG_ID, PG_NAME, PG_NPI
    PG_ID = pg_id
    PG_NAME = pg_name
    PG_NPI = pg_npi
    
    print(f"""
🔄 PHYSICIAN GROUP CONFIGURATION UPDATED:
   📋 PG Name: {PG_NAME}
   🆔 PG ID: {PG_ID}  
   🔢 PG NPI: {PG_NPI}
""")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TOKEN = os.getenv("AUTH_TOKEN")
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
DOC_STATUS_URL = "https://api.doctoralliance.com/document/get?docId.id="
PATIENT_CREATE_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/create"
ORDER_PUSH_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Order"

# =============================================================================
# PHYSICIAN GROUP (PG) CONFIGURATION
# =============================================================================
# Configure these parameters for the specific PG you want to process
# Update these values for different physician groups:
# 
# Example for Hawthorn Medical Associates:
# PG_ID = "4b51c8b7-c8c4-4779-808c-038c057f026b"  
# PG_NAME = "Hawthorn Medical Associates - Family Practice"
# PG_NPI = "1932207917"
#
# To use for a different PG, simply change these 3 values:
# =============================================================================
PG_ID = "d074279d-8ff6-47ab-b340-04f21c0f587e"
PG_NAME = "Aco Health Solutions" 
PG_NPI = "1306161443"

print(f"""
🏥 PHYSICIAN GROUP CONFIGURATION ACTIVE:
   📋 PG Name: {PG_NAME}
   🆔 PG ID: {PG_ID}  
   🔢 PG NPI: {PG_NPI}
""")

# =============================================================================
# AGENCY MAPPING CONFIGURATION
# =============================================================================
# The system now uses the facility name from the CSV as the agency name
# and maps it to the agency ID using the company_map loaded from output.json
# =============================================================================

# Load company mapping from output.json - maps agency names to IDs
with open("output.json") as f:
    company_map = json.load(f)

# Load or initialize created patients cache
if os.path.exists("created_patients.json") and os.path.getsize("created_patients.json") > 0:
    with open("created_patients.json") as f:
        created_patients = json.load(f)
else:
    created_patients = {}

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
AUDIT_PATIENTS_FILE = "audit_patients.csv"
AUDIT_ORDERS_FILE = "audit_orders.csv"

# --- YOLO+DocTR CONFIG ---
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.3
TEMP_DIR = "temp_crops"
os.makedirs(TEMP_DIR, exist_ok=True)
DOCTR_MODEL = ocr_predictor(pretrained=True, assume_straight_pages=True)
YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
YOLO_MODEL.to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
YOLO_LOCK = threading.Lock()

def is_scanned_pdf(pdf_bytes):
    """Heuristic: if most pages have no text, it's scanned."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        no_text_pages = sum(1 for page in pdf.pages if not safe_strip(page.extract_text() or ''))
        logger.info("Document appears to be scanned - using OCR pipeline")
        return no_text_pages > (len(pdf.pages) // 2)

def extract_text_pdfplumber(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        logger.info("Document is text-based - using pdfplumber for extraction")
    return text

def enhance_image(img: Image.Image) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=5, searchWindowSize=15)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    return Image.fromarray(enhanced).convert('RGB')

def yolo_doctr_extract(pdf_bytes):
    """Extract text from scanned PDF using YOLO+DocTR pipeline."""
    temp_pdf_path = os.path.join(TEMP_DIR, "temp_input.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
    doc = fitz.open(temp_pdf_path)
    all_text = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        zoom = 3
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = enhance_image(img)
        temp_img_path = os.path.join(TEMP_DIR, f"page_{page_num}.png")
        img.save(temp_img_path, "PNG")
        # YOLO detection
        with YOLO_LOCK:
            results = YOLO_MODEL(source=temp_img_path, verbose=False, conf=YOLO_CONFIDENCE, imgsz=640)
        coords = []
        for result in results:
            if result.boxes is not None:
                coords.extend(result.boxes.xyxy.cpu().numpy())
        # If regions found, crop and OCR each
        if coords:
            for i, bbox in enumerate(coords):
                xmin, ymin, xmax, ymax = bbox / zoom
                rect = fitz.Rect(xmin, ymin, xmax, ymax)
                cropped_pix = page.get_pixmap(clip=rect, matrix=matrix, alpha=False, colorspace=fitz.csRGB)
                cropped_img = Image.frombytes("RGB", [cropped_pix.width, cropped_pix.height], cropped_pix.samples)
                crop_path = os.path.join(TEMP_DIR, f"crop_{page_num}_{i}.png")
                cropped_img.save(crop_path, "PNG")
                # DocTR OCR
                doc_img = DocumentFile.from_images(crop_path)
                result = DOCTR_MODEL(doc_img)
                text_content = ""
                for pg in result.pages:
                    for block in pg.blocks:
                        for line in block.lines:
                            line_text = " ".join(word.value for word in line.value)
                            if safe_strip(line_text):
                                text_content += safe_strip(line_text) + "\n"
                all_text.append(text_content)
                os.remove(crop_path)
        else:
            # fallback: full page tesseract
            text = pytesseract.image_to_string(temp_img_path)
            all_text.append(text)
        os.remove(temp_img_path)
    doc.close()
    os.remove(temp_pdf_path)
    return "\n".join(all_text)

def get_pdf_text(doc_id):
    try:
        # Try to fetch document with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS, timeout=30)
                if response.status_code == 200:
                    break
                elif attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch document after {max_retries} attempts. Status: {response.status_code}")
                else:
                    logger.warning(f"Document fetch attempt {attempt + 1} failed with status {response.status_code}, retrying...", doc_id)
                    time.sleep(2)  # Wait before retry
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception(f"Document fetch timed out after {max_retries} attempts")
                else:
                    logger.warning(f"Document fetch timeout attempt {attempt + 1}, retrying...", doc_id)
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Network error after {max_retries} attempts: {str(e)}")
                else:
                    logger.warning(f"Network error attempt {attempt + 1}: {str(e)}, retrying...", doc_id)
                    time.sleep(2)
        
        # Validate response structure
        try:
            doc_data = response.json()
            if not doc_data or "value" not in doc_data:
                raise Exception("Invalid response format: missing 'value' key")
            
            value_data = doc_data["value"]
            if not value_data:
                raise Exception("Invalid response format: empty 'value' data")
            
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from document API")
        
        # Extract document metadata with fallbacks
        try:
            # Required fields
            if "patientId" not in value_data or not value_data["patientId"]:
                raise Exception("Missing required patientId in document")
            
            daId = value_data["patientId"]["id"] if isinstance(value_data["patientId"], dict) else value_data["patientId"]
            
            if "documentBuffer" not in value_data or not value_data["documentBuffer"]:
                raise Exception("Missing required documentBuffer in document")
            
            document_buffer = value_data["documentBuffer"]
            
            # Optional fields with defaults
            is_faxed = value_data.get("isFaxed", False)
            fax_source = value_data.get("faxSource", "Unknown")
            document_type = value_data.get("documentType", "Unknown")
            
        except (KeyError, TypeError) as e:
            raise Exception(f"Error extracting document metadata: {str(e)}")
        
        # Decode PDF with error handling
        try:
            pdf_bytes = base64.b64decode(document_buffer)
            if len(pdf_bytes) == 0:
                raise Exception("Empty PDF document buffer")
            
        except Exception as e:
            raise Exception(f"Error decoding PDF document: {str(e)}")
        
        # Extract text with multiple fallback methods
        text = ""
        extraction_method = "unknown"
        
        try:
            # Method 1: Try pdfplumber first (fastest for text-based PDFs)
            try:
                text = extract_text_pdfplumber(pdf_bytes)
                extraction_method = "pdfplumber"
                if text and text.strip():
                    logger.success(f"Text extracted successfully using {extraction_method}", doc_id)
                else:
                    raise Exception("pdfplumber extracted empty text")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {str(e)}, trying OCR...", doc_id)
                
                # Method 2: Try YOLO+DocTR OCR
                try:
                    text = yolo_doctr_extract(pdf_bytes)
                    extraction_method = "yolo_doctr"
                    if text and text.strip():
                        logger.success(f"Text extracted successfully using {extraction_method}", doc_id)
                    else:
                        raise Exception("YOLO+DocTR extracted empty text")
                except Exception as e:
                    logger.warning(f"YOLO+DocTR extraction failed: {str(e)}, trying basic OCR...", doc_id)
                    
                    # Method 3: Fallback to basic Tesseract OCR
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                            temp_file.write(pdf_bytes)
                            temp_file.flush()
                            
                            # Convert PDF to images and OCR
                            import fitz
                            doc = fitz.open(temp_file.name)
                            text_parts = []
                            
                            for page_num in range(min(doc.page_count, 10)):  # Limit to first 10 pages
                                page = doc[page_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                                img_data = pix.tobytes("png")
                                
                                # OCR the image
                                import pytesseract
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(img_data))
                                page_text = pytesseract.image_to_string(img, config='--psm 6')
                                text_parts.append(page_text)
                            
                            doc.close()
                            os.unlink(temp_file.name)
                            
                            text = "\n".join(text_parts)
                            extraction_method = "tesseract_fallback"
                            
                            if text and text.strip():
                                logger.success(f"Text extracted successfully using {extraction_method}", doc_id)
                            else:
                                raise Exception("All OCR methods extracted empty text")
                                
                    except Exception as e:
                        logger.error(f"All text extraction methods failed: {str(e)}", doc_id)
                        # Last resort: return minimal text to prevent complete failure
                        text = f"[EXTRACTION_FAILED] Document ID: {doc_id}"
                        extraction_method = "error_handling"
                        logger.warning("Text extraction failed completely - document will be skipped", doc_id)
        
        except Exception as e:
            logger.error(f"Critical error in text extraction: {str(e)}", doc_id)
            text = f"[CRITICAL_ERROR] Document ID: {doc_id}"
            extraction_method = "error_placeholder"
        
        # Clean up extracted text
        try:
            if text and safe_strip(text):
                # Remove specific patterns that might cause issues
                edited = re.sub(r'\b\d[A-Z][A-Z0-9]\d[A-Z][A-Z0-9]\d[A-Z]{2}(?:\d{2})?\b', '', text)
                # Remove excessive whitespace
                edited = re.sub(r'\s+', ' ', edited).strip()
            else:
                edited = text
        except Exception as e:
            logger.warning(f"Text cleaning failed: {str(e)}, using raw text", doc_id)
            edited = text
        
        # Log extraction summary
        logger.info(f"Text extraction completed using {extraction_method}: {len(edited)} characters", doc_id)
        
        # Return text, daId, and metadata
        return [
            edited, 
            daId, 
            {
                "isFaxed": is_faxed, 
                "faxSource": fax_source, 
                "documentType": document_type,
                "extractionMethod": extraction_method,
                "textLength": len(edited)
            }
        ]
        
    except Exception as e:
        logger.error(f"Critical failure in get_pdf_text: {str(e)}", doc_id)
        # SKIP PROCESSING - No fallback data creation
        logger.warning(f"Document {doc_id} will be skipped due to text extraction failure", doc_id)
        return None

def extract_patient_data(text):
    try:
        # Validate input text
        if not text or not safe_strip(text):
            logger.warning("Empty or invalid text provided for patient data extraction")
            return {}
        
        # If text contains error markers, try to extract what we can
        if "[EXTRACTION_FAILED]" in text or "[CRITICAL_ERROR]" in text:
            logger.warning("Text extraction had issues, attempting basic pattern matching")
            # Try basic regex patterns for common fields
            extracted_data = {}
            
            # Basic name patterns
            name_patterns = [
                r'(?:Patient|Name)[\s:]+([A-Z][a-z]+)\s+([A-Z][a-z]+)',
                r'([A-Z][a-z]+),?\s+([A-Z][a-z]+)',
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted_data["patientFName"] = match.group(1).strip()
                    extracted_data["patientLName"] = match.group(2).strip()
                    break
            
            # Basic DOB patterns
            dob_patterns = [
                r'(?:DOB|Date.?of.?Birth)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ]
            
            for pattern in dob_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted_data["dob"] = match.group(1).strip()
                    break
            
            # If we found any data, return it with empty fields for the rest
            if extracted_data:
                logger.info("Extracted basic patient data using pattern matching")
                full_data = {
                    "patientFName": extracted_data.get("patientFName", ""),
                    "patientLName": extracted_data.get("patientLName", ""),
                    "dob": extracted_data.get("dob", ""),
                    "patientSex": "",
                    "medicalRecordNo": "",
                    "billingProvider": "",
                    "physicianNPI": "",
                    "nameOfAgency": "",
                    "patientAddress": "",
                    "patientCity": "",
                    "patientState": "",
                    "zip": "",
                    "episodeDiagnoses": [
                        {
                            "startOfCare": "",
                            "startOfEpisode": "",
                            "endOfEpisode": "",
                            "firstDiagnosis": "",
                            "secondDiagnosis": "",
                            "thirdDiagnosis": "",
                            "fourthDiagnosis": "",
                            "fifthDiagnosis": "",
                            "sixthDiagnosis": ""
                        }
                    ]
                }
                return full_data
        
        query = """
        You are an expert in medical documentation. Extract the following fields from the attached medical PDF and return them in the specified JSON format. Use the exact field names below, and provide values based strictly on the document content. This JSON will be sent directly to an API, so ensure the format is correct and do not return any additional text.
        
        Required JSON format:
        {
        "patientFName": "",
        "patientLName": "",
        "dob": "",
        "patientSex": "",
        "medicalRecordNo": "",
        "billingProvider": "",
        "physicianNPI": "",
        "nameOfAgency": "",
        "patientAddress": "",
        "patientCity": "",
        "patientState": "",
        "zip": "",
        "episodeDiagnoses": [
            {
            "startOfCare": "",
            "startOfEpisode": "",
            "endOfEpisode": "",
            "firstDiagnosis": "",
            "secondDiagnosis": "",
            "thirdDiagnosis": "",
            "fourthDiagnosis": "",
            "fifthDiagnosis": "",
            "sixthDiagnosis": ""
            }
        ]
        }
        
        CRITICAL EXTRACTION RULES:
        
        **ABSOLUTELY FORBIDDEN PATTERNS:**
        - NEVER create patient names starting with "Unknown_" followed by any numbers
        - NEVER create patient names like "PATIENT_9284559" or "GENERATED_PATIENT"
        - NEVER generate synthetic patient names when names are not found
        - If you cannot find patient names, leave patientFName and patientLName as empty strings ""
        - DO NOT make up, infer, or create fallback patient names under any circumstances
        
        DATE FORMATTING:
        - ALL dates must be in MM/DD/YYYY format (American standard)
        - Convert any date format to MM/DD/YYYY (e.g., "4/6/2025" → "04/06/2025", "2025-04-06" → "04/06/2025")
        - Pad single digits with zeros (e.g., "4/6/25" → "04/06/2025", assume 20XX for 2-digit years)
        - If date is unclear or invalid, leave field empty
        
        NPI CLEANING:
        - Extract only the numeric NPI value (10 digits)
        - Remove any formatting: brackets [], parentheses (), decimal points and trailing zeros (.0)
        - Example: "[1234567890.0]" → "1234567890"
        - Example: "(1234567890)" → "1234567890"
        - If NPI is not exactly 10 digits after cleaning, leave field empty
        
        FIELD-SPECIFIC INSTRUCTIONS:
        
        Patient Name:
        - patientFName: First name only, no middle names or initials
        - patientLName: Last name only, no suffixes (Jr., Sr., etc.)
        - Look for labels: "Patient Name", "Patient", "Name", or similar
        - CRITICAL: If no patient name is found, return empty strings "", do NOT create "Unknown_" patterns
        
        Date of Birth (dob):
        - Look for labels: "DOB", "Date of Birth", "Birth Date", "Born"
        - Convert to MM/DD/YYYY format
        
        Patient Sex:
        - Look for: "Sex", "Gender", "M/F"
        - Return only: "M", "F", "Male", "Female" (standardize to single letter if possible)
        
        Medical Record Number:
        - Look for: "MRN", "Medical Record", "Record Number", "Patient ID"
        - Extract alphanumeric value only
        
        Billing Provider:
        - Primary physician name responsible for billing
        - Look for: "Physician", "Provider", "Attending", "Doctor"
        - Return full name as written
        
        Physician NPI:
        - Look for: "NPI", "Provider NPI", "Physician NPI", "National Provider"
        - Apply NPI cleaning rules above
        - Must be exactly 10 digits after cleaning
        
        Name of Agency:
        - Healthcare agency/organization name
        - Look for: "Agency", "Organization", "Facility", "Provider Organization"
        
        Patient Address Fields:
        - patientAddress: Street address only (number and street name)
          Look for: "Address", "Street", "Patient Address", "Home Address"
          Example: "237 WOOD AVE" (exclude city, state, zip)
        - patientCity: City name only
          Look for: "City", "Patient City", labels near address
          Example: "HYDE PARK"
        - patientState: State abbreviation (2 letters) or full state name
          Look for: "State", "ST", "Patient State", labels near address
          Example: "MA" or "Massachusetts"
        - zip: ZIP code only (5 digits or 5+4 format)
          Look for: "ZIP", "Zip Code", "Postal Code", numeric codes near address
          Example: "2136" or "02136" or "02136-1234"
        
        EPISODE DATES:
        - startOfCare: Look for "SOC", "Start of Care", "SOC Date", "Care Start"
        - startOfEpisode: Look for "Start Date", "Episode Start", "Certification From", "Period From"
          If in "X - Y" format, use date before the dash
        - endOfEpisode: Look for "End Date", "Episode End", "Certification To", "Period To"
          If in "X - Y" format, use date after the dash
        - All dates in MM/DD/YYYY format
        
        DIAGNOSIS CODES:
        - Extract up to 6 ICD-10-CM codes (primary + secondary)
        - Format: Letter + 2 digits + optional dot + up to 4 alphanumerics
        - Examples: "M25.511", "Z51.11", "I50.9"
        - Look for: "Diagnosis", "ICD-10", "Primary Diagnosis", "Secondary Diagnosis"
        - Return only the code, not descriptions
        
        EXTRACTION PRIORITY:
        1. Look for explicitly labeled fields first
        2. Search for common medical form patterns
        3. Use contextual clues (headers, sections, formatting)
        4. Do NOT infer or guess values
        5. Leave fields empty if not found or unclear
        6. NEVER create "Unknown_" or artificial fallback values
        
        If you cannot extract certain fields, leave them as empty strings rather than making up values.
        Return ONLY the JSON object with no additional text, comments, or explanations.
        """
        
        # Try AI extraction with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
                if not response or not response.text:
                    raise Exception("Empty response from AI model")
                
                # Try to extract JSON from response
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if not match:
                    raise Exception("No JSON found in AI response")
                
                json_str = match.group()
                extracted_data = json.loads(json_str)
                
                # Validate extracted data structure
                required_fields = ["patientFName", "patientLName", "dob", "episodeDiagnoses"]
                for field in required_fields:
                    if field not in extracted_data:
                        logger.warning(f"Missing required field: {field}, adding empty value")
                        if field == "episodeDiagnoses":
                            extracted_data[field] = [{"startOfCare": "", "startOfEpisode": "", "endOfEpisode": "", "firstDiagnosis": "", "secondDiagnosis": "", "thirdDiagnosis": "", "fourthDiagnosis": "", "fifthDiagnosis": "", "sixthDiagnosis": ""}]
                        else:
                            extracted_data[field] = ""
                
                # CRITICAL: Detect and reject Unknown patterns from AI
                fname = safe_strip(extracted_data.get("patientFName", "")).upper()
                lname = safe_strip(extracted_data.get("patientLName", "")).upper()
                
                if fname.startswith("UNKNOWN_") or lname.startswith("UNKNOWN_") or \
                   "UNKNOWN_" in fname or "UNKNOWN_" in lname:
                    logger.error("AI generated forbidden Unknown patient pattern - rejecting extraction")
                    logger.error(f"Rejected name: '{fname}' '{lname}'")
                    # Return empty structure instead of Unknown patterns
                    return {
                        "patientFName": "",
                        "patientLName": "",
                        "dob": "",
                        "patientSex": "",
                        "medicalRecordNo": "",
                        "billingProvider": "",
                        "physicianNPI": "",
                        "nameOfAgency": "",
                        "patientAddress": "",
                        "patientCity": "",
                        "patientState": "",
                        "zip": "",
                        "episodeDiagnoses": [
                            {
                                "startOfCare": "",
                                "startOfEpisode": "",
                                "endOfEpisode": "",
                                "firstDiagnosis": "",
                                "secondDiagnosis": "",
                                "thirdDiagnosis": "",
                                "fourthDiagnosis": "",
                                "fifthDiagnosis": "",
                                "sixthDiagnosis": ""
                            }
                        ]
                    }
                
                # Add any missing optional fields
                optional_fields = ["patientSex", "medicalRecordNo", "billingProvider", "physicianNPI", "nameOfAgency", "patientAddress", "patientCity", "patientState", "zip"]
                for field in optional_fields:
                    if field not in extracted_data:
                        extracted_data[field] = ""
                
                logger.success("AI patient data extraction completed successfully")
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All JSON parsing attempts failed")
                else:
                    time.sleep(1)  # Brief pause before retry
                    
            except Exception as e:
                error_message = str(e).lower()
                if "api key" in error_message or "authentication" in error_message or "expired" in error_message:
                    logger.error("The Gemini API key has expired or is invalid. Please update your API key in the .env file.")
                    logger.error("To fix this:")
                    logger.error("1. Go to https://aistudio.google.com/apikey")
                    logger.error("2. Generate a new API key")
                    logger.error("3. Update the GEMINI_API_KEY in your .env file")
                    raise Exception("Gemini API key expired or invalid. Please update your API key.")
                else:
                    logger.warning(f"AI extraction attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error("All AI extraction attempts failed")
                    else:
                        time.sleep(1)  # Brief pause before retry
        
        # If AI extraction completely fails, return minimal structure
        logger.warning("AI extraction failed, returning minimal patient data structure")
        return {
            "patientFName": "",
            "patientLName": "",
            "dob": "",
            "patientSex": "",
            "medicalRecordNo": "",
            "billingProvider": "",
            "physicianNPI": "",
            "nameOfAgency": "",
            "patientAddress": "",
            "patientCity": "",
            "patientState": "",
            "zip": "",
            "episodeDiagnoses": [
                {
                    "startOfCare": "",
                    "startOfEpisode": "",
                    "endOfEpisode": "",
                    "firstDiagnosis": "",
                    "secondDiagnosis": "",
                    "thirdDiagnosis": "",
                    "fourthDiagnosis": "",
                    "fifthDiagnosis": "",
                    "sixthDiagnosis": ""
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Critical error in extract_patient_data: {str(e)}")
        # Return minimal structure to prevent complete failure
        return {
            "patientFName": "",
            "patientLName": "",
            "dob": "",
            "patientSex": "",
            "medicalRecordNo": "",
            "billingProvider": "",
            "physicianNPI": "",
            "nameOfAgency": "",
            "patientAddress": "",
            "patientCity": "",
            "patientState": "",
            "zip": "",
            "episodeDiagnoses": [
                {
                    "startOfCare": "",
                    "startOfEpisode": "",
                    "endOfEpisode": "",
                    "firstDiagnosis": "",
                    "secondDiagnosis": "",
                    "thirdDiagnosis": "",
                    "fourthDiagnosis": "",
                    "fifthDiagnosis": "",
                    "sixthDiagnosis": ""
                }
            ]
        }

def extract_order_data(text):
    try:
        # Validate input text
        if not text or not safe_strip(text):
            logger.warning("Empty or invalid text provided for order data extraction")
            return {}
        
        # If text contains error markers, try basic pattern matching
        if "[EXTRACTION_FAILED]" in text or "[CRITICAL_ERROR]" in text:
            logger.warning("Text extraction had issues, attempting basic order pattern matching")
            return {
                "orderNo": "",
                "orderDate": "",
                "startOfCare": "",
                "episodeStartDate": "",
                "episodeEndDate": "",
                "documentID": "",
                "mrn": "",
                "patientName": "",
                "sentToPhysicianDate": "",
                "sentToPhysicianStatus": False,
                "patientId": "",
                "companyId": "",
                "bit64Url": "",
                "documentName": ""
            }
        
        query = """
        You are an expert in medical documentation. Extract the following fields from the attached medical PDF and return them in the specified JSON format. Use the exact field names below, and provide values based strictly on the document content. This JSON will be sent directly to an API, so ensure the format is correct and do not return any additional text.
    
        Required JSON format:
        {
        "orderNo": "",
        "orderDate": "",
        "startOfCare": "",
        "episodeStartDate": "",
        "episodeEndDate": "",
        "documentID": "",
        "mrn": "",
        "patientName": "",
        "sentToPhysicianDate": "",
        "sentToPhysicianStatus": false,
        "patientId": "",
        "companyId": "",
        "bit64Url": "",
        "documentName": ""
        }

        CRITICAL EXTRACTION RULES:

        DATE FORMATTING:
        - ALL dates must be in MM/DD/YYYY format (American standard)
        - Convert any date format to MM/DD/YYYY (e.g., "4/6/2025" → "04/06/2025", "2025-04-06" → "04/06/2025")
        - Pad single digits with zeros (e.g., "4/6/25" → "04/06/2025", assume 20XX for 2-digit years)
        - If date is unclear or invalid, leave field empty

        FIELD-SPECIFIC INSTRUCTIONS:

        Order Number (orderNo):
        - Look for: "Order #", "Order No", "Order Number", "Reference #", "Document #"
        - Extract alphanumeric value only
        - Should be a meaningful identifier, not random text

        Order Date (orderDate):
        - Look for: "Order Date", "Date Ordered", "Created Date", "Document Date"
        - Convert to MM/DD/YYYY format

        Start of Care (startOfCare):
        - Look for: "SOC", "Start of Care", "SOC Date", "Care Start Date"
        - Convert to MM/DD/YYYY format

        Episode Start Date (episodeStartDate):
        - Look for: "Start Date", "Episode Start", "Certification From", "Period From", "From Date"
        - If in "X - Y" date range format, use the date BEFORE the dash
        - Convert to MM/DD/YYYY format

        Episode End Date (episodeEndDate):
        - Look for: "End Date", "Episode End", "Certification To", "Period To", "To Date"
        - If in "X - Y" date range format, use the date AFTER the dash
        - Convert to MM/DD/YYYY format
        - Should be 59 or 89 days after episode start date

        Document ID (documentID):
        - Look for: "Document ID", "Doc ID", "ID", unique document identifier
        - Extract alphanumeric value only

        Medical Record Number (mrn):
        - Look for: "MRN", "Medical Record Number", "Patient MRN", "Record #", "Medical Record #"
        - VALIDATION RULES:
          * Must be alphanumeric (letters and numbers only)
          * Length should be between 4-20 characters
          * Should NOT be random text, dates, or irrelevant numbers
          * Common formats: "REC123456", "12345", "ABC123", etc.
          * If value seems random or invalid, leave field empty
        - Do NOT extract: phone numbers, SSN, dates, or unrelated numeric sequences

        Patient Name (patientName):
        - Look for: "Patient Name", "Patient", "Name"
        - Return full name as written (First Last format preferred)

        Sent to Physician Date (sentToPhysicianDate):
        - Look for: "Sent Date", "Physician Date", "Submitted Date"
        - Convert to MM/DD/YYYY format

        Sent to Physician Status (sentToPhysicianStatus):
        - Look for indicators of whether document was sent to physician
        - Return boolean: true or false (not string)

        IMPORTANT NOTES:
        - Leave empty strings for fields that cannot be found
        - Do NOT guess or infer values
        - Return only the JSON with no additional text
        
        If you cannot extract certain fields, leave them as empty strings or appropriate defaults rather than making up values.
        Return ONLY the JSON object with no additional text, comments, or explanations.
        """
        
        # Try AI extraction with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
                if not response or not response.text:
                    raise Exception("Empty response from AI model")
                
                # Try to extract JSON from response
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if not match:
                    raise Exception("No JSON found in AI response")
                
                json_str = match.group()
                extracted_data = json.loads(json_str)
                
                # Validate and ensure all required fields exist
                required_fields = {
                    "orderNo": "",
                    "orderDate": "",
                    "startOfCare": "",
                    "episodeStartDate": "",
                    "episodeEndDate": "",
                    "documentID": "",
                    "mrn": "",
                    "patientName": "",
                    "sentToPhysicianDate": "",
                    "sentToPhysicianStatus": False,
                    "patientId": "",
                    "companyId": "",
                    "bit64Url": "",
                    "documentName": ""
                }
                
                # Add missing fields with defaults
                for field, default_value in required_fields.items():
                    if field not in extracted_data:
                        logger.warning(f"Missing order field: {field}, adding default value")
                        extracted_data[field] = default_value
                
                # Validate sentToPhysicianStatus is boolean
                if not isinstance(extracted_data.get("sentToPhysicianStatus"), bool):
                    extracted_data["sentToPhysicianStatus"] = False
                
                logger.success("AI order data extraction completed successfully")
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"Order JSON parsing failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All order JSON parsing attempts failed")
                else:
                    time.sleep(1)  # Brief pause before retry
                    
            except Exception as e:
                error_message = str(e).lower()
                if "api key" in error_message or "authentication" in error_message or "expired" in error_message:
                    logger.error("The Gemini API key has expired or is invalid for order extraction.")
                    raise Exception("Gemini API key expired or invalid. Please update your API key.")
                else:
                    logger.warning(f"AI order extraction attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error("All AI order extraction attempts failed")
                    else:
                        time.sleep(1)  # Brief pause before retry
        
        # If AI extraction completely fails, return minimal structure
        logger.warning("AI order extraction failed, returning minimal order data structure")
        return {
            "orderNo": "",
            "orderDate": "",
            "startOfCare": "",
            "episodeStartDate": "",
            "episodeEndDate": "",
            "documentID": "",
            "mrn": "",
            "patientName": "",
            "sentToPhysicianDate": "",
            "sentToPhysicianStatus": False,
            "patientId": "",
            "companyId": "",
            "bit64Url": "",
            "documentName": ""
        }
        
    except Exception as e:
        logger.error(f"Critical error in extract_order_data: {str(e)}")
        # Return minimal structure to prevent complete failure
        return {
            "orderNo": "",
            "orderDate": "",
            "startOfCare": "",
            "episodeStartDate": "",
            "episodeEndDate": "",
            "documentID": "",
            "mrn": "",
            "patientName": "",
            "sentToPhysicianDate": "",
            "sentToPhysicianStatus": False,
            "patientId": "",
            "companyId": "",
            "bit64Url": "",
            "documentName": ""
        }

def fetch_signed_date(doc_id):
    response = requests.get(f"{DOC_STATUS_URL}{doc_id}", headers=HEADERS)
    if response.status_code == 200:
        value = response.json().get("value", {})
        if value.get("documentStatus") == "Signed":
            raw_date = value.get("physicianSigndate", "")
            try:
                return datetime.fromisoformat(raw_date).strftime("%m/%d/%Y") if raw_date else None
            except ValueError:
                return None
    return None

def get_patient_details_from_api(patient_id):
    try:
        response = requests.get(f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/getPatient?id={patient_id}", headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("data")
        else:
            return None
    except Exception:
        logger.warning(f"Failed to fetch patient details for patient ID: {patient_id}")
        return None


def process_dates_for_patient(patient_data, doc_id, audit_reason=None):
    # Always ensure episode_info dict exists so function never returns None
    try:
        episode_info = patient_data.get("episodeDiagnoses", [{}])[0] or {}
    except Exception:
        episode_info = {}
        patient_data["episodeDiagnoses"] = [episode_info]
        # Log missing episode diagnosis but continue processing
        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, "No Episode Diagnosis"])

    def safe_parse_date(date_str):
        try:
            date_str = safe_strip(date_str)
            if not date_str:
                return None
            return datetime.strptime(date_str, "%m/%d/%Y")
        except Exception:
            return None

    def format_date(dt):
        return dt.strftime("%m/%d/%Y") if dt else ""

    soc = episode_info.get("startOfCare", "")
    soe = episode_info.get("startOfEpisode", "")
    eoe = episode_info.get("endOfEpisode", "")

    soc_dt = safe_parse_date(soc)
    soe_dt = safe_parse_date(soe)
    eoe_dt = safe_parse_date(eoe)

    logger.data("Raw Episode Dates from Document", {
        "Start of Care": soc or "Not found",
        "Start of Episode": soe or "Not found", 
        "End of Episode": eoe or "Not found"
    }, doc_id)

    # Fill missing SOC
    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    # Fill missing SOE
    if not soe_dt and eoe_dt:
        soe_dt = eoe_dt - timedelta(days=59)

    # Fill missing EOE
    if not eoe_dt and soe_dt:
        eoe_dt = soe_dt + timedelta(days=59)

    # If only SOC exists
    if not soe_dt and not eoe_dt and soc_dt:
        soe_dt = soc_dt
        eoe_dt = soc_dt + timedelta(days=59)

    # If still none of the dates are available
    if not soc_dt and not soe_dt and not eoe_dt:
        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, "Missing SOC, SOE, EOE"])
        # Even if dates are missing, return patient_data unchanged to avoid NoneType errors
        return patient_data

    # Ensure all dates are formatted correctly at the end
    episode_info["startOfCare"] = format_date(soc_dt)
    episode_info["startOfEpisode"] = format_date(soe_dt)
    episode_info["endOfEpisode"] = format_date(eoe_dt)

    patient_data["episodeDiagnoses"][0] = episode_info

    logger.data("Parsed Episode Dates", {
        "Start of Care": soc_dt.strftime("%m/%d/%Y") if soc_dt else "Not found",
        "Start of Episode": soe_dt.strftime("%m/%d/%Y") if soe_dt else "Not found", 
        "End of Episode": eoe_dt.strftime("%m/%d/%Y") if eoe_dt else "Not found"
    }, doc_id)

    return patient_data

from datetime import datetime, timedelta

def process_dates_for_order(order_data, patient_id):
    def parse_date(d):
        try:
            return datetime.strptime(d, "%m/%d/%Y") if d else None
        except ValueError:
            return None

    def format_date(d):
        return d.strftime("%m/%d/%Y") if d else ""

    # Extract existing order dates
    soc = order_data.get("startOfCare")
    soe = order_data.get("episodeStartDate")
    eoe = order_data.get("episodeEndDate")

    soc_dt = parse_date(soc)
    soe_dt = parse_date(soe)
    eoe_dt = parse_date(eoe)

    patient_details = get_patient_details_from_api(patient_id)
    agency_info = patient_details.get("agencyInfo", {}) if patient_details else {}

    if not soc_dt and not soe_dt and not eoe_dt:
        logger.info("Using Start of Care date to derive episode dates")
        soe_dt = parse_date(agency_info.get("startOfCare"))
        eoe_dt = soe_dt + timedelta(days=59) if soe_dt else None

    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    # Case 2: Compute missing SOE from EOE
    if not soe_dt and eoe_dt:
        soe_dt = eoe_dt - timedelta(days=59)

    # Case 3: Compute missing EOE from SOE
    if not eoe_dt and soe_dt:
        eoe_dt = soe_dt + timedelta(days=59)
        
    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    order_data["startOfCare"] = format_date(soc_dt)
    order_data["episodeStartDate"] = format_date(soe_dt)
    order_data["episodeEndDate"] = format_date(eoe_dt)

    logger.data("Retrieved Agency Info", {
        "Start of Care": agency_info.get("startOfCare", "Not available"),
        "Start of Episode": soe_dt.strftime("%m/%d/%Y") if soe_dt else "Not available",
        "End of Episode": eoe_dt.strftime("%m/%d/%Y") if eoe_dt else "Not available"
    })

    return order_data


def check_if_patient_exists(fname, lname, dob, agency_id):
    """Check if patient exists in the specific agency or all agencies if agency_id is None"""
    dob = safe_strip(dob)
    fname = safe_strip(fname).upper()
    lname = safe_strip(lname).upper()

    check_ids = [
        PG_ID
    ]

    # If agency_id is provided, search only that agency first
    if agency_id:
        check_ids = [agency_id] + [id for id in check_ids if id != agency_id]

    for check_id in check_ids:
        url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/company/{check_id}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                if response.status_code == 404:
                    logger.debug(f"No patients in agency ID: {check_id} (Status: {response.status_code})")
                else:
                    logger.info(f"Agency ID: {check_id} returned status: {response.status_code}")
                continue

            try:
                patients = response.json()
            except Exception as e:
                logger.warning(f"Error parsing patient list JSON for agency ID {check_id}: {e}")
                continue

            for p in patients:
                info = p.get("agencyInfo", {})
                if not info:
                    continue
                existing_fname = safe_strip(info.get("patientFName")).upper()
                existing_lname = safe_strip(info.get("patientLName")).upper()
                existing_dob = safe_strip(info.get("dob"))
                if existing_fname == fname and existing_lname == lname and existing_dob == dob:
                    patient_id = p.get("id") or p.get("patientId")
                    logger.success(f"✅ Patient found in system | {fname} {lname} (DOB: {dob}) | Agency: {check_id} | Patient ID: {patient_id}")
                    return patient_id

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout when checking patients for agency ID: {check_id}")
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error when checking patients for agency ID {check_id}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error for agency ID {check_id}: {e}")
            continue

    return None

def get_or_create_patient(patient_data, daId, agency):
    logger.data("Patient Data for Processing", {
        "First Name": patient_data.get("patientFName", ""),
        "Last Name": patient_data.get("patientLName", ""),
        "DOB": patient_data.get("dob", ""),
        "MRN": patient_data.get("medicalRecordNo", ""),
        "Address": patient_data.get("patientAddress", ""),
        "City": patient_data.get("patientCity", ""),
        "State": patient_data.get("patientState", ""),
        "ZIP": patient_data.get("zip", "")
    })
    
    dob = safe_strip(patient_data.get("dob"))
    fname = safe_strip(patient_data.get("patientFName")).upper()
    lname = safe_strip(patient_data.get("patientLName")).upper()

    # Skip if no valid patient identifiers
    if not fname or not lname or not dob:
        logger.error("Missing required patient identifiers - skipping document")
        return None

    key = f"{fname}_{lname}_{dob}"

    # Get agency ID from company mapping first
    agency_id = company_map.get(safe_strip(agency).lower())
    if not agency_id:
        logger.error(f"Agency '{agency}' not found in company mapping")
        return None

    # Check locally created patients
    if key in created_patients:
        logger.success(f"Patient already processed in this session: {fname} {lname}")
        return created_patients[key]

    # Check if patient exists in API for this specific agency
    existing_id = check_if_patient_exists(fname, lname, dob, agency_id)
    if existing_id:
        logger.success(f"Patient found on platform: {fname} {lname}, ID: {existing_id}")
        created_patients[key] = existing_id  # Cache it to avoid refetching
        with open("created_patients.json", "w") as f:
            json.dump(created_patients, f, indent=2)
        return existing_id

    # Proceed to create new patient
    patient_data["companyId"] = agency_id
    patient_data["nameOfAgency"] = safe_strip(agency)
    
    # Add PG-specific fields
    patient_data["daBackofficeID"] = str(daId)
    patient_data["pgCompanyId"] = PG_ID
    patient_data["physicianGroup"] = PG_NAME
    patient_data["physicianGroupNPI"] = PG_NPI
    
    logger.progress("Creating new patient...")
    logger.data("Patient Creation Request", patient_data)
    
    resp = requests.post(PATIENT_CREATE_URL, headers={"Content-Type": "application/json"}, json=patient_data)
    
    logger.info(f"Patient creation response - Status: {resp.status_code}")
    if resp.status_code == 201:
        new_id = resp.json().get("id") or resp.text
        created_patients[key] = new_id
        with open("created_patients.json", "w") as f:
            json.dump(created_patients, f, indent=2)
        logger.success(f"New patient created successfully with ID: {new_id}")
        return new_id
    elif resp.status_code == 409:
        # Patient already exists - search across all agencies
        logger.success(f"Patient already exists on platform: {fname} {lname}")
        logger.info("Searching for existing patient to retrieve their ID...")
        
        existing_id = check_if_patient_exists(fname, lname, dob, None)  # Search all agencies
        if existing_id:
            logger.success(f"Successfully found existing patient: {fname} {lname}, ID: {existing_id}")
            created_patients[key] = existing_id
            with open("created_patients.json", "w") as f:
                json.dump(created_patients, f, indent=2)
            return existing_id
        else:
            logger.warning(f"Patient exists (409) but could not locate their ID in comprehensive search")
            logger.info(f"API Response: {resp.text}")
            return None
    else:
        logger.error(f"Failed to create patient - Status: {resp.status_code}")
        logger.error(f"Response: {resp.text}")
        return None

def push_order(order_data, doc_id):
    logger.progress("Preparing order for submission", doc_id)
    logger.data("Order Request Data", {
        "Order No": order_data.get("orderNo", "Not set"),
        "Patient ID": order_data.get("patientId", "Not set"),
        "Company ID": order_data.get("companyId", "Not set"),
        "Order Date": order_data.get("orderDate", "Not set"),
        "Start of Care": order_data.get("startOfCare", "Not set"),
        "Episode Start": order_data.get("episodeStartDate", "Not set"),
        "Episode End": order_data.get("episodeEndDate", "Not set")
    }, doc_id)
    
    # Ensure order number exists and is non-empty
    if not order_data.get("orderNo") or not safe_strip(order_data.get("orderNo")):
        logger.error(f"No valid order number found - cannot process order", doc_id)
        return "Order number missing", 400
    
    resp = requests.post(ORDER_PUSH_URL, headers={"Content-Type": "application/json"}, json=order_data)
    
    try:
        new_resp = resp.json()  # Parse the response to a dict
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response from order push", doc_id)
        logger.error(f"Raw response: {resp.text}", doc_id)
        return resp.text, resp.status_code
    
    order_id = new_resp.get("orderId")
    
    if resp.status_code == 201:
        logger.success(f"Order pushed successfully with ID: {order_id}", doc_id)
    elif resp.status_code == 409:
        # Order already exists - this is often a successful outcome
        logger.success(f"Order already exists in system with ID: {order_id}", doc_id)
        logger.info("This indicates the order was processed successfully in a previous run", doc_id)
    elif resp.status_code == 400:
        # Bad request - could be duplicate order or validation issue
        logger.warning(f"Order submission issue - Status: {resp.status_code}", doc_id)
        logger.info(f"Response: {resp.text}", doc_id)
        with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, "Validation Issue", order_id or "N/A"])
    else:
        logger.error(f"Order push failed - Status: {resp.status_code}", doc_id)
        with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, f"Failed - Status {resp.status_code}", order_id or "N/A"])
        logger.error(f"Response: {resp.text}", doc_id)

    return resp.text, resp.status_code


def remove_none_fields(data):
    """Remove fields with None values from dictionary"""
    return {k: v for k, v in data.items() if v is not None}

def synchronize_mrn(patient_data, order_data, doc_id):
    """
    Synchronize MRN between patient and order data
    Priority: 
    1. Use extracted MRN if valid and present in either patient or order data
    2. SKIP PROCESSING if no valid MRN found (NO GENERATION)
    """
    patient_mrn = safe_strip(patient_data.get("medicalRecordNo"))
    order_mrn = safe_strip(order_data.get("mrn"))
    
    # Function to validate MRN
    def is_valid_mrn(mrn):
        if not mrn:
            return False
        # Check length (4-20 characters)
        if len(mrn) < 4 or len(mrn) > 20:
            return False
        # Check if alphanumeric
        if not mrn.isalnum():
            return False
        # Check if it's not just a date or phone number pattern
        if mrn.isdigit() and len(mrn) in [8, 10]:  # Likely date or phone
            return False
        return True
    
    # Determine which MRN to use
    final_mrn = None
    mrn_source = "none"
    
    if is_valid_mrn(patient_mrn):
        final_mrn = patient_mrn
        mrn_source = "patient_extracted"
    elif is_valid_mrn(order_mrn):
        final_mrn = order_mrn
        mrn_source = "order_extracted"
    else:
        # NO FALLBACK - Return None to indicate processing should be skipped
        logger.warning(f"No valid MRN found - document will be skipped", doc_id)
        logger.data("MRN Validation Failed", {
            "Patient MRN (extracted)": patient_mrn or "None",
            "Order MRN (extracted)": order_mrn or "None", 
            "Validation": "FAILED - No valid MRN found",
            "Action": "Document will be skipped"
        }, doc_id)
        return None, None
    
    # Apply the final MRN to both patient and order data
    patient_data["medicalRecordNo"] = final_mrn
    order_data["mrn"] = final_mrn
    
    logger.data("MRN Synchronization", {
        "Patient MRN (extracted)": patient_mrn or "None",
        "Order MRN (extracted)": order_mrn or "None", 
        "Final MRN": final_mrn,
        "Source": mrn_source,
        "Applied to": "Both patient and order"
    }, doc_id)
    
    return patient_data, order_data

# === SELENIUM NPI EXTRACTION FUNCTION ===
def fetch_physician_npi_selenium(doc_id):
    """
    Fetch physician NPI using selenium automation for accurate extraction
    Returns the NPI number or None if extraction fails
    """
    driver = None
    try:
        logger.info(f"Starting selenium NPI extraction for document ID: {doc_id}", doc_id)
        
        # Set up Chrome options for headless operation
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--start-maximized')
        
        # Initialize the driver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        wait = WebDriverWait(driver, 20)
        
        # Login to Doctor Alliance
        logger.progress("Logging into Doctor Alliance...", doc_id)
        driver.get("https://backoffice.doctoralliance.com/")
        time.sleep(3)
        
        # Login credentials - try multiple selectors
        try:
            username_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']")))
            password_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
        except TimeoutException:
            logger.warning("Trying alternative login selectors...", doc_id)
            username_element = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Username']")))
            password_element = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Password']")))
        
        username_value = os.getenv("DA_USERNAME")
        password_value = os.getenv("DA_PASSWORD")
        
        # Clear and enter credentials with delays (more reliable)
        username_element.clear()
        password_element.clear()
        
        for char in username_value:
            username_element.send_keys(char)
            time.sleep(0.05)
        
        for char in password_value:
            password_element.send_keys(char)
            time.sleep(0.05)
        
        # Submit login - try multiple button selectors
        try:
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        except TimeoutException:
            logger.warning("Trying alternative login button selector...", doc_id)
            login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.login-button")))
        
        driver.execute_script("arguments[0].click();", login_button)
        time.sleep(3)
        
        # Navigate to search - with better error handling
        logger.progress("Navigating to search section...", doc_id)
        time.sleep(2)  # Wait for sidebar to load
        
        try:
            search_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Search')]")))
        except TimeoutException:
            logger.warning("Trying alternative search link selectors...", doc_id)
            try:
                search_link = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='Search']")))
            except TimeoutException:
                search_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'Search')]")))
        
        driver.execute_script("arguments[0].click();", search_link)
        time.sleep(2)
        
        # Perform search with document ID
        logger.progress(f"Searching for document ID: {doc_id}", doc_id)
        search_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Enter search text...']")))
        search_input.clear()
        
        # Enter document ID with delay (more reliable)
        for char in doc_id:
            search_input.send_keys(char)
            time.sleep(0.05)
        
        # Select Documents from dropdown with better handling
        logger.progress("Setting search type to Documents...", doc_id)
        try:
            dropdown = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".select2-selection")))
            dropdown.click()
            time.sleep(1)
            
            documents_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[contains(@class, 'select2-results__option') and contains(text(), 'Documents')]")))
            documents_option.click()
            time.sleep(1)
        except TimeoutException:
            logger.warning("Dropdown selection failed, trying JavaScript alternative...", doc_id)
            driver.execute_script("""
                var select = document.querySelector('select[name="SearchType"]');
                if(select) {
                    select.value = 'Documents';
                    var event = new Event('change', { bubbles: true });
                    select.dispatchEvent(event);
                }
            """)
            time.sleep(1)
        
        # Submit search
        search_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Search')]")))
        search_button.click()
        
        # Wait for results and click on document
        logger.progress("Waiting for search results...", doc_id)
        time.sleep(3)
        
        # Click on the document row - more robust approach
        try:
            rows = driver.find_elements(By.XPATH, "//table[contains(@class, 'table')]//tbody//tr")
            clicked = False
            
            for row in rows:
                try:
                    id_cell = row.find_element(By.XPATH, ".//td[1]")
                    row_id = safe_strip(id_cell.text)
                    
                    if row_id == str(doc_id):
                        logger.progress(f"Found document row with ID: {row_id}, clicking...", doc_id)
                        driver.execute_script("arguments[0].click();", id_cell)
                        clicked = True
                        break
                except:
                    continue
                    
            if not clicked:
                # Fallback: click first row if specific ID not found
                logger.warning("Specific document ID not found, clicking first result...", doc_id)
                first_row = wait.until(EC.element_to_be_clickable((By.XPATH, "//table[contains(@class, 'table')]//tbody//tr[1]")))
                driver.execute_script("arguments[0].click();", first_row)
                
        except Exception as e:
            logger.error(f"Failed to click document row: {str(e)}", doc_id)
            return None
        
        time.sleep(3)  # Wait for document page to load
        
        # Extract NPI from document page - comprehensive approach
        logger.progress("Extracting physician NPI from document...", doc_id)
        
        # Debug: Save page source for analysis
        try:
            with open(f"debug_page_{doc_id}.html", "w", encoding='utf-8') as f:
                f.write(driver.page_source)
            logger.info(f"Page source saved to debug_page_{doc_id}.html for analysis", doc_id)
        except Exception as e:
            logger.warning(f"Could not save debug page: {str(e)}", doc_id)
        
        # Method 1: Look for NPI in visible elements
        try:
            # Get all text on the page for debugging
            body_text = driver.find_element(By.TAG_NAME, "body").text
            logger.info(f"Page contains {len(body_text)} characters of visible text", doc_id)
            
            # Look for any mention of NPI, physician names, or numbers in brackets
            all_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'NPI') or contains(text(), 'Physician') or contains(text(), 'Dr.') or contains(text(), 'Andrade') or contains(text(), '[') or contains(text(), 'Eric') or contains(text(), 'Levine')]")
            
            logger.info(f"Found {len(all_elements)} elements containing relevant text", doc_id)
            
            for i, element in enumerate(all_elements[:10]):  # Limit to first 10 for debugging
                element_text = safe_strip(element.text)
                element_html = element.get_attribute('innerHTML')
                logger.info(f"Element {i+1} text: {element_text[:100]}...", doc_id)
                
                for content in [element_text, element_html]:
                    # Try various NPI patterns
                    patterns = [
                        r'\[(\d{10})\]',  # 10-digit NPI in brackets
                        r'\[(\d{8,12})\]',  # 8-12 digit numbers in brackets
                        r'NPI[:\s]*(\d{10})',  # NPI: followed by 10 digits
                        r'NPI[:\s]*(\d{8,12})',  # NPI: followed by 8-12 digits
                        r'(\d{10})',  # Any 10-digit number
                        r'(\d{8,12})'  # Any 8-12 digit number
                    ]
                    
                    for pattern in patterns:
                        npi_match = re.search(pattern, content)
                        if npi_match:
                            npi = npi_match.group(1)
                            if len(npi) >= 8:
                                logger.success(f"Found NPI in element using pattern {pattern}: {npi}", doc_id)
                                return npi
        except Exception as e:
            logger.warning(f"Element-based NPI extraction failed: {str(e)}", doc_id)
        
        # Method 2: Look in page source with comprehensive patterns
        try:
            page_source = driver.page_source
            logger.info(f"Page source contains {len(page_source)} characters", doc_id)
            
            # Save a snippet of page source for debugging
            snippet = page_source[:2000] + "..." if len(page_source) > 2000 else page_source
            logger.info(f"Page source snippet: {snippet}", doc_id)
            
            # Comprehensive NPI patterns
            npi_patterns = [
                # Specific physician patterns
                r'Dr\.\s*Andrade[^[]*\[(\d+)\]',
                r'Eric\s+Levine[^[]*\[(\d+)\]',
                r'Levine[^[]*\[(\d+)\]',
                r'Andrade[^[]*\[(\d+)\]',
                
                # General patterns
                r'Physician[^[]*\[(\d+)\]',
                r'Dr\.[^[]*\[(\d+)\]',
                r'NPI[:\s]*(\d{8,12})',
                r'npi[:\s]*(\d{8,12})',
                
                # Bracket patterns
                r'\[(\d{10})\]',  # 10-digit NPI in brackets
                r'\[(\d{8,12})\]',  # 8-12 digit numbers in brackets
                
                # Any physician ID patterns
                r'physicianId["\']?\s*:\s*["\']?(\d+)["\']?',
                r'physician_id["\']?\s*:\s*["\']?(\d+)["\']?',
                
                # JSON-like patterns
                r'"physician"[^}]*"id"[^:]*:\s*["\']?(\d+)["\']?',
                r'"id"[^:]*:\s*["\']?(\d{8,12})["\']?'
            ]
            
            for pattern in npi_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE | re.DOTALL)
                if matches:
                    for match in matches:
                        npi = str(match)
                        if len(npi) >= 8:  # NPI should be at least 8 digits
                            logger.success(f"Found NPI using pattern {pattern}: {npi}", doc_id)
                            return npi
                            
        except Exception as e:
            logger.warning(f"Page source NPI extraction failed: {str(e)}", doc_id)
        
        # Method 3: Look in JavaScript and data attributes
        try:
            # Execute JavaScript to find any data containing numbers
            js_data = driver.execute_script("""
                var allData = [];
                
                // Look for data attributes
                var elements = document.querySelectorAll('[data-*]');
                for(var i = 0; i < elements.length; i++) {
                    var attrs = elements[i].attributes;
                    for(var j = 0; j < attrs.length; j++) {
                        if(attrs[j].name.startsWith('data-') && /\\d{8,}/.test(attrs[j].value)) {
                            allData.push(attrs[j].name + ': ' + attrs[j].value);
                        }
                    }
                }
                
                // Look for JavaScript variables
                var scripts = document.querySelectorAll('script');
                for(var i = 0; i < scripts.length; i++) {
                    var content = scripts[i].textContent || scripts[i].innerText;
                    if(content && /\\d{8,}/.test(content)) {
                        allData.push('Script content with numbers found');
                    }
                }
                
                return allData;
            """)
            
            logger.info(f"JavaScript data search found: {js_data}", doc_id)
            
            # Try to extract NPI from any found data
            for data_item in js_data:
                npi_matches = re.findall(r'(\d{8,12})', str(data_item))
                for npi in npi_matches:
                    if len(npi) >= 8:
                        logger.success(f"Found NPI in JavaScript data: {npi}", doc_id)
                        return npi
                        
        except Exception as e:
            logger.warning(f"JavaScript NPI extraction failed: {str(e)}", doc_id)
        
        # Method 4: Look for any table data that might contain physician info
        try:
            tables = driver.find_elements(By.TAG_NAME, "table")
            logger.info(f"Found {len(tables)} tables on the page", doc_id)
            
            for i, table in enumerate(tables):
                table_text = table.text
                logger.info(f"Table {i+1} content preview: {table_text[:200]}...", doc_id)
                
                # Look for NPI patterns in table text
                npi_patterns = [r'\[(\d{8,12})\]', r'(\d{10})', r'NPI[:\s]*(\d{8,12})']
                for pattern in npi_patterns:
                    matches = re.findall(pattern, table_text, re.IGNORECASE)
                    for match in matches:
                        npi = str(match)
                        if len(npi) >= 8:
                            logger.success(f"Found NPI in table {i+1}: {npi}", doc_id)
                            return npi
                            
        except Exception as e:
            logger.warning(f"Table NPI extraction failed: {str(e)}", doc_id)
        
        # Method 5: Get all text and look for any numeric patterns
        try:
            all_text = driver.find_element(By.TAG_NAME, "body").text
            logger.info(f"Searching through all visible text ({len(all_text)} chars)", doc_id)
            
            # Find all 8-12 digit numbers
            all_numbers = re.findall(r'\b(\d{8,12})\b', all_text)
            logger.info(f"Found potential numbers: {all_numbers}", doc_id)
            
            # Filter for likely NPI numbers (typically 10 digits)
            for number in all_numbers:
                if len(number) == 10:
                    logger.success(f"Found 10-digit number (potential NPI): {number}", doc_id)
                    return number
                    
            # If no 10-digit, try other lengths
            for number in all_numbers:
                if len(number) >= 8:
                    logger.success(f"Found {len(number)}-digit number (potential NPI): {number}", doc_id)
                    return number
                    
        except Exception as e:
            logger.warning(f"Full text NPI extraction failed: {str(e)}", doc_id)
        
        logger.warning("Could not extract NPI using selenium", doc_id)
        return None
        
    except Exception as e:
        logger.error(f"Selenium NPI extraction failed: {str(e)}", doc_id)
        return None
        
    finally:
        if driver:
            driver.quit()

def process_csv(csv_path):
    def normalize_date_string(date_str):
        try:
            # Replace hyphens with slashes if needed
            date_str = safe_strip(date_str, "").replace("-", "/")
            # Parse using flexible format
            dt = datetime.strptime(date_str, "%m/%d/%Y")
            # Format with leading zeroes
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return ""
    
    logger.header("PATIENT DATA PIPELINE - PROCESSING STARTED")
    logger.info(f"Processing CSV file: {csv_path}")
    
    # Initialize detailed counters
    total_documents = 0
    processed_documents = 0
    failed_documents = 0
    skipped_documents = 0
    successful_patients = 0
    successful_orders = 0
    patient_errors = 0
    order_errors = 0
    
    try:
        i = 0
        with open(csv_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # Count total documents first
            total_documents = sum(1 for _ in reader)
            file.seek(0)  # Reset file pointer
            next(reader)  # Skip header row
            
            for row in reader:
                # Remove test limit - process all documents
                if i > 49:
                    break
                i += 1
                doc_id = row["ID"]
                agency = safe_strip(row.get("Facility"))
                received = normalize_date_string(safe_strip(row.get("Received On")))
                
                logger.header(f"PROCESSING DOCUMENT {doc_id}")
                logger.info(f"Facility/Agency: {agency} | Received Date: {received}", doc_id)
                
                # Track success for this document
                document_success = True
                patient_success = False
                order_success = False
                
                try:
                    # Check if agency exists in mapping
                    agency_id = company_map.get(agency.lower())
                    if not agency_id:
                        logger.warning(f"Agency '{agency}' not found in company mapping", doc_id)
                        skipped_documents += 1
                        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Agency '{agency}' not found in mapping", "SKIPPED"])
                        continue  # Skip to next document

                    # Step 1: Extract text from PDF (with robust error handling)
                    logger.step(1, 6, "Extracting text from PDF", doc_id)
                    text = ""
                    daId = ""
                    doc_metadata = {}
                    
                    try:
                        res = get_pdf_text(doc_id)
                        
                        # Check if text extraction failed (returns None)
                        if res is None:
                            logger.error("Text extraction failed - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Text extraction failed", "SKIPPED"])
                            continue  # Skip to next document
                            
                        text = res[0]
                        daId = res[1]
                        doc_metadata = res[2]
                        
                        # Validate text extraction
                        if not text or not safe_strip(text):
                            logger.warning("Empty text extracted - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Empty text extracted", "SKIPPED"])
                            continue  # Skip to next document
                        
                        if len(safe_strip(text)) < 50:
                            logger.warning(f"Very short text extracted ({len(text)} chars) - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Text too short for processing", "SKIPPED"])
                            continue  # Skip to next document
                        
                        logger.success("Text extraction completed", doc_id)
                        
                        # Display fax status prominently
                        logger.fax_status(
                            is_faxed=doc_metadata.get("isFaxed", False),
                            fax_source=doc_metadata.get("faxSource", "Unknown"),
                            doc_type=doc_metadata.get("documentType", "Unknown"),
                            doc_id=doc_id
                        )
                        
                        logger.data("Document Information", {
                            "Faxed": "Yes" if doc_metadata.get("isFaxed", False) else "No",
                            "Fax Source": doc_metadata.get("faxSource", "N/A"), 
                            "Document Type": doc_metadata.get("documentType", "Unknown"),
                            "Extraction Method": doc_metadata.get("extractionMethod", "Unknown"),
                            "Text Length": doc_metadata.get("textLength", len(text))
                        }, doc_id)
                        
                    except Exception as e:
                        logger.error(f"PDF extraction failed: {str(e)} - skipping document", doc_id)
                        skipped_documents += 1
                        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"PDF extraction failed: {str(e)}", "SKIPPED"])
                        continue  # Skip to next document

                    # Step 2: Extract and process patient data (with validation)
                    logger.step(2, 6, "Extracting patient data using AI", doc_id)
                    patient_data = {}
                    
                    try:
                        patient_data = extract_patient_data(text)
                        if not patient_data:
                            logger.warning("No patient data extracted - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "No patient data extracted", "SKIPPED"])
                            continue  # Skip to next document
                        
                        logger.data("AI-Extracted Patient Data", patient_data, doc_id)
                        
                        # Validate critical patient fields
                        has_name = bool(safe_strip(patient_data.get("patientFName")) or safe_strip(patient_data.get("patientLName")))
                        has_dob = bool(safe_strip(patient_data.get("dob")))
                        
                        if not has_name or not has_dob:
                            logger.error("Missing critical patient identifiers - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Missing patient identifiers (name and DOB)", "SKIPPED"])
                            continue  # Skip to next document
                        
                    except Exception as e:
                        logger.error(f"Patient data extraction failed: {str(e)} - skipping document", doc_id)
                        skipped_documents += 1
                        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Extraction failed: {str(e)}", "SKIPPED"])
                        continue  # Skip to next document

                    # Step 3: Process dates with better validation
                    logger.step(3, 6, "Processing and validating dates", doc_id)
                    try:
                        patient_data = process_dates_for_patient(patient_data, doc_id)
                        if not patient_data:
                            logger.error("Date processing failed - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Date processing failed", "SKIPPED"])
                            continue  # Skip to next document
                        logger.data("Patient Data After Date Processing", patient_data, doc_id)
                    except Exception as e:
                        logger.error(f"Date processing failed: {str(e)} - skipping document", doc_id)
                        skipped_documents += 1
                        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Date processing failed: {str(e)}", "SKIPPED"])
                        continue  # Skip to next document

                    # Step 4: Get or create patient (with fallbacks)
                    logger.step(4, 6, "Creating or finding patient", doc_id)
                    patient_id = None
                    
                    try:
                        patient_id = get_or_create_patient(patient_data, daId, agency)
                        
                        if not patient_id:
                            logger.error("Patient processing failed - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Patient creation/retrieval failed", "SKIPPED"])
                            continue  # Skip to next document
                            
                        patient_success = True
                        successful_patients += 1
                        logger.success(f"Patient processed successfully with ID: {patient_id}", doc_id)
                            
                    except Exception as e:
                        logger.error(f"Patient creation/retrieval failed: {str(e)} - skipping document", doc_id)
                        skipped_documents += 1
                        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Patient processing error: {str(e)}", "SKIPPED"])
                        continue  # Skip to next document

                    # Step 5: Extract and process order data
                    logger.step(5, 6, "Extracting order data", doc_id)
                    order_data = {}
                    
                    try:
                        order_data = extract_order_data(text)
                        if not order_data:
                            logger.warning("No order data extracted - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "No order data extracted", "SKIPPED"])
                            continue  # Skip to next document
                        
                        logger.data("AI-Extracted Order Data", order_data, doc_id)
                        
                    except Exception as e:
                        logger.error(f"Order data extraction failed: {str(e)} - skipping document", doc_id)
                        skipped_documents += 1
                        with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Order extraction failed: {str(e)}", "SKIPPED"])
                        continue  # Skip to next document

                    # Step 6: Process order with enhanced validation
                    logger.step(6, 6, "Processing order submission", doc_id)
                    try:
                        # Process order dates
                        try:
                            order_data = process_dates_for_order(order_data, doc_id)
                            if not order_data:
                                logger.error("Order date processing failed - skipping document", doc_id)
                                skipped_documents += 1
                                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                    writer = csv.writer(file)
                                    writer.writerow([doc_id, "Order date processing failed", "SKIPPED"])
                                continue  # Skip to next document
                            logger.data("Order Data After Date Processing", order_data, doc_id)
                        except Exception as e:
                            logger.error(f"Order date processing failed: {str(e)} - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, f"Order date processing failed: {str(e)}", "SKIPPED"])
                            continue  # Skip to next document
                        
                        # Synchronize MRN between patient and order
                        try:
                            patient_data, order_data = synchronize_mrn(patient_data, order_data, doc_id)
                            
                            # Check if MRN synchronization failed (returns None, None)
                            if patient_data is None or order_data is None:
                                logger.error("Document skipped due to missing valid MRN", doc_id)
                                skipped_documents += 1
                                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                    writer = csv.writer(file)
                                    writer.writerow([doc_id, "Missing valid MRN", "SKIPPED"])
                                continue  # Skip to next document
                                
                        except Exception as e:
                            logger.error(f"MRN synchronization failed: {str(e)} - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, f"MRN sync failed: {str(e)}", "SKIPPED"])
                            continue  # Skip to next document
                        
                        # Validate required fields
                        if not agency_id:
                            logger.error("No agency ID available for order processing - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "No agency ID available", "SKIPPED"])
                            continue  # Skip to next document
                        
                        # Set required order fields
                        order_data["companyId"] = agency_id
                        order_data["pgCompanyId"] = PG_ID
                        order_data["patientId"] = patient_id
                        order_data["documentID"] = doc_id
                        
                        # Validate dates exist for order submission
                        required_dates = ["orderDate", "episodeStartDate", "episodeEndDate"]
                        missing_dates = [date for date in required_dates if not safe_strip(order_data.get(date))]
                        
                        if missing_dates:
                            logger.error(f"Missing required dates: {missing_dates} - skipping document", doc_id)
                            skipped_documents += 1
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, f"Missing required dates: {missing_dates}", "SKIPPED"])
                            continue  # Skip to next document
                        
                        # Remove None values before submission
                        order_data = remove_none_fields(order_data)
                        
                        # Attempt order submission
                        try:
                            response_text, status_code = push_order(order_data, doc_id)
                            
                            # Consider various status codes as successful
                            if status_code in [201, 409]:  # Created or Conflict (already exists)
                                order_success = True
                                successful_orders += 1
                                logger.success("Order processing completed successfully", doc_id)
                            elif status_code == 400:
                                # Bad request might still be partially successful
                                order_success = True  # Consider it successful since we got a response
                                successful_orders += 1
                                logger.info("Order had validation issues but was processed", doc_id)
                            else:
                                order_success = False
                                order_errors += 1
                                logger.error(f"Order submission failed with status: {status_code}", doc_id)
                                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                    writer = csv.writer(file)
                                    writer.writerow([doc_id, f"Order submission failed: {status_code}", "SKIPPED"])
                        except Exception as e:
                            logger.error(f"Order submission failed: {str(e)}", doc_id)
                            order_success = False
                            order_errors += 1
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, f"Order submission failed: {str(e)}", "SKIPPED"])
                        
                    except Exception as e:
                        logger.error(f"Order processing failed: {str(e)}", doc_id)
                        order_success = False
                        order_errors += 1
                        with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Order processing failed: {str(e)}", "SKIPPED"])

                    # Document is successful if either patient or order succeeded
                    if patient_success or order_success:
                        logger.success("Document processing completed successfully!", doc_id)
                        processed_documents += 1
                    else:
                        logger.warning("Document processing completed with issues", doc_id)
                        failed_documents += 1
                    
                except Exception as e:
                    logger.error(f"Critical error processing document {doc_id}: {str(e)}")
                    failed_documents += 1
                    
                    # Log critical failure
                    with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, f"Critical failure: {str(e)}", "SKIPPED"])
                    
                    # Continue to next document instead of stopping
                    continue

        # Display comprehensive final statistics
        logger.header("PROCESSING STATISTICS")
        
        # Calculate rates
        overall_success_rate = (processed_documents/total_documents*100) if total_documents > 0 else 0
        patient_success_rate = (successful_patients/total_documents*100) if total_documents > 0 else 0
        order_success_rate = (successful_orders/total_documents*100) if total_documents > 0 else 0
        skip_rate = (skipped_documents/total_documents*100) if total_documents > 0 else 0
        
        logger.data("Overall Document Processing", {
            "Total Documents": total_documents,
            "Successfully Processed": processed_documents,
            "Skipped Documents": skipped_documents,
            "Failed Documents": failed_documents,
            "Overall Success Rate": f"{overall_success_rate:.1f}%",
            "Skip Rate": f"{skip_rate:.1f}%"
        })
        
        logger.data("Patient Processing Details", {
            "Successful Patients": successful_patients,
            "Patient Errors": patient_errors,
            "Patient Success Rate": f"{patient_success_rate:.1f}%",
            "Note": "Documents with missing patient data are now skipped instead of creating unknown patients"
        })
        
        logger.data("Order Processing Details", {
            "Successful Orders": successful_orders,
            "Order Errors": order_errors,
            "Order Success Rate": f"{order_success_rate:.1f}%",
            "Note": "Includes new orders, duplicate orders, and successful submissions"
        })
        
        # Success indicators
        if overall_success_rate >= 90:
            logger.success(f"🎉 EXCELLENT SUCCESS RATE: {overall_success_rate:.1f}%")
        elif overall_success_rate >= 70:
            logger.success(f"✅ GOOD SUCCESS RATE: {overall_success_rate:.1f}%")
        elif overall_success_rate >= 50:
            logger.warning(f"⚠️  MODERATE SUCCESS RATE: {overall_success_rate:.1f}%")
        else:
            logger.error(f"❌ LOW SUCCESS RATE: {overall_success_rate:.1f}% - Review logs for issues")
        
        logger.header("PATIENT DATA PIPELINE - PROCESSING COMPLETED")
    finally:
        logger.close()

def main():
    """
    Main execution function with usage examples
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    PHYSICIAN GROUP PROCESSOR                 ║
║             Enhanced Patient & Order Processing               ║
╚══════════════════════════════════════════════════════════════╝

📋 USAGE EXAMPLES:

1. Default Configuration (Hawthorn Medical Associates):
   python final.py

2. Custom PG Configuration:
   configure_physician_group(
       pg_id="your-pg-id",
       pg_name="Your Physician Group Name", 
       pg_npi="your-pg-npi"
   )

3. CSV Processing Options:
   - orders.csv (default)
   - signed.csv  
   - orders_3.csv
   - custom file path

💡 TIP: Edit the PG configuration constants at the top of this file
    for permanent changes, or use configure_physician_group() for
    runtime changes.
""")
    
    # Start processing with default CSV
    process_csv("hawthorn_fam.csv")

if __name__ == "__main__":
    main()
else:
    # Direct call for backwards compatibility
    process_csv("hawthorn_fam.csv")