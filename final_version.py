import csv
import json
import base64
import io
import re
import os
import time
from datetime import datetime, timedelta
import requests
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv

# OCR imports for fallback
import cv2
import numpy as np
import fitz
from PIL import Image
import pytesseract

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
TOKEN = os.getenv("AUTH_TOKEN")
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
DOC_STATUS_URL = "https://api.doctoralliance.com/document/get?docId.id="
PATIENT_CREATE_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/create"
ORDER_PUSH_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Order"
PG_ID = "4b51c8b7-c8c4-4779-808c-038c057f026b"  
PG_NAME = "Hawthorn Medical Associates - Family Practice"
PG_NPI = "1508825811"

# Load mappings
with open("output.json") as f:
    company_map = json.load(f)

if os.path.exists("hawthorn_internalmedicine.json") and os.path.getsize("hawthorn_internalmedicine.json") > 0:
    with open("hawthorn_internalmedicine.json") as f:
        created_patients = json.load(f)
else:
    created_patients = {}

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
AUDIT_PATIENTS_FILE = "audit_patients.csv"
AUDIT_ORDERS_FILE = "audit_orders.csv"

# Create directories for outputs
os.makedirs("api_outputs", exist_ok=True)
os.makedirs("csv_outputs", exist_ok=True)

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

# === UTILITY FUNCTIONS ===
def safe_strip(value, default=""):
    """Safely strip a value, handling None and non-string types"""
    try:
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return default

def remove_none_fields(data):
    """Remove None fields from data but preserve required date fields"""
    required_date_fields = {'orderDate', 'startOfCare', 'episodeStartDate', 'episodeEndDate'}
    if isinstance(data, dict):
        return {
            k: remove_none_fields(v)
            for k, v in data.items()
            if v is not None or k in required_date_fields
        }
    elif isinstance(data, list):
        return [remove_none_fields(item) for item in data]
    elif data is None:
        return ""
    return data

def write_to_csv(patient_data, order_data, doc_id, agency, csv_writer):
    """Write patient and order data to CSV file"""
    try:
        # Get episode data
        episode_info = patient_data.get("episodeDiagnoses", [{}])[0] if patient_data.get("episodeDiagnoses") else {}
        
        # Calculate age from date of birth
        age = ""
        dob_str = patient_data.get("dob", "")
        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%m/%d/%Y")
                today = datetime.now()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                age = str(age)
                logger.info(f"Calculated age: {age} from DOB: {dob_str}", doc_id)
            except ValueError:
                logger.warning(f"Could not parse DOB for age calculation: {dob_str}", doc_id)
                age = ""
        
        # Convert M/F to MALE/FEMALE for output
        sex_value = patient_data.get("patientSex", "")
        if sex_value == "M":
            sex_display = "MALE"
        elif sex_value == "F":
            sex_display = "FEMALE"
        else:
            sex_display = sex_value  # Keep as is if already MALE/FEMALE or empty
        
        # Extract service line and payer source
        service_line = patient_data.get("serviceLine", "")
        payer_source = patient_data.get("payorSource", "")
        
        # Log the extracted values for these critical fields
        logger.info(f"Service Line extracted: '{service_line}'", doc_id)
        logger.info(f"Payer Source extracted: '{payer_source}'", doc_id)
        
        # Log diagnosis codes being written
        logger.info(f"Diagnosis codes being written:", doc_id)
        for i in range(1, 7):
            dx_field = f"{'first' if i==1 else 'second' if i==2 else 'third' if i==3 else 'fourth' if i==4 else 'fifth' if i==5 else 'sixth'}Diagnosis"
            dx_value = episode_info.get(dx_field, "")
            logger.info(f"  - Latest_Episode_{dx_field.title()}: '{dx_value}'", doc_id)
        
        # Create row data matching the expected CSV format
        row_data = {
            'ID': '',  # Will be generated
            'Created At': datetime.now().strftime("%m/%d/%Y"),
            'Created By': 'FINAL_VERSION_PROCESSOR',
            'Is Billable': 'TRUE',
            'Is PG Billable': 'TRUE', 
            'Is Eligible': 'TRUE',
            'Is PG Eligible': 'TRUE',
            'Patient WAV ID': f"PAT{doc_id}",
            'Patient EHR Record ID': '',  # Don't enter MRN here
            'Patient EHR Type': 'Kantime',
            'First Name': patient_data.get("patientFName", ""),
            'Middle Name': '',
            'Last Name': patient_data.get("patientLName", ""),
            'Profile Picture': '',
            'Date of Birth': patient_data.get("dob", ""),
            'Age': age,  # Now calculated from DOB
            'Sex': sex_display,
            'Status': 'Active',
            'Marital Status': '',
            'SSN': '',
            'Start of Care': episode_info.get("startOfCare", ""),
            'Medical Record No': patient_data.get("medicalRecordNo", ""),
            'Service Line': service_line,  # Now extracted from patient data
            'Address': patient_data.get("address", ""),
            'City': patient_data.get("city", ""),
            'State': patient_data.get("state", ""),
            'Zip': patient_data.get("zip", ""),
            'Email': patient_data.get("email", ""),
            'Phone Number': patient_data.get("phoneNumber", ""),
            'Fax': '',
            'Payor Source': payer_source,  # Now extracted from patient data
            'Billing Provider': patient_data.get("billingProvider", ""),
            'Billing Provider Phone': '',
            'Billing Provider Address': '',
            'Billing Provider Zip': '',
            'NPI': '',  # Don't pass anything in the NPI column
            'Line1 DOS From': order_data.get("episodeStartDate", ""),
            'Line1 DOS To': order_data.get("episodeEndDate", ""),
            'Line1 POS': '',
            'Physician NPI': patient_data.get("physicianNPI", ""),  # Attending/Primary physician NPI
            'Supervising Provider': '',
            'Supervising Provider NPI': '',
            'Physician Group': PG_NAME,
            'Physician Group NPI': PG_NPI,
            'Physician Group Address': '',
            'Physician Phone': '',
            'Physician Address': '',
            'City State Zip': '',
            'Patient Account No': '',
            'Agency NPI': '',
            'Name of Agency': patient_data.get("nameOfAgency", agency),
            'Insurance ID': '',
            'Primary Insurance': '',
            'Secondary Insurance': '',
            'Secondary Insurance ID': '',
            'Tertiary Insurance': '',
            'Tertiary Insurance ID': '',
            'Next of Kin': '',
            'Patient Caretaker': '',
            'Caretaker Contact Number': '',
            'Remarks': f"Extracted from document {doc_id}",
            'DA Backoffice ID': doc_id,
            'Company ID': company_map.get(agency.strip().lower(), ""),
            'PG Company ID': PG_ID,
            'Signed Orders': '1',
            'Total Orders': '1',
            'Latest_Episode_StartOfCare': episode_info.get("startOfCare", ""),
            'Latest_Episode_StartOfEpisode': episode_info.get("startOfEpisode", ""),
            'Latest_Episode_EndOfEpisode': episode_info.get("endOfEpisode", ""),
            'Latest_Episode_FirstDiagnosis': episode_info.get("firstDiagnosis", ""),
            'Latest_Episode_SecondDiagnosis': episode_info.get("secondDiagnosis", ""),
            'Latest_Episode_ThirdDiagnosis': episode_info.get("thirdDiagnosis", ""),
            'Latest_Episode_FourthDiagnosis': episode_info.get("fourthDiagnosis", ""),
            'Latest_Episode_FifthDiagnosis': episode_info.get("fifthDiagnosis", ""),
            'Latest_Episode_SixthDiagnosis': episode_info.get("sixthDiagnosis", "")
        }
        
        # Write the row
        csv_writer.writerow(row_data)
        logger.success(f"Patient data written to CSV: {patient_data.get('patientFName', '')} {patient_data.get('patientLName', '')} | Sex: {sex_display} | Age: {age} | Service: {service_line} | Payer: {payer_source} | MRN: {patient_data.get('medicalRecordNo', 'N/A')}", doc_id)
        return True
        
    except Exception as e:
        logger.error(f"Error writing to CSV: {str(e)}", doc_id)
        return False

# === TEXT EXTRACTION METHODS ===

def extract_text_pdfplumber(pdf_bytes):
    """Primary text extraction method using pdfplumber"""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
    except Exception as e:
        raise Exception(f"PDFPlumber extraction failed: {str(e)}")

def extract_text_tesseract_ocr(pdf_bytes):
    """Fallback OCR method using Tesseract"""
    try:
        # Convert PDF to images and OCR
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        
        for page_num in range(min(doc.page_count, 10)):  # Limit to first 10 pages
            page = doc[page_num]
            # Higher resolution for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # OCR the image
            img = Image.open(io.BytesIO(img_data))
            # Convert to grayscale for better OCR
            img = img.convert('L')
            page_text = pytesseract.image_to_string(img, config='--psm 6')
            if page_text.strip():
                text_parts.append(page_text)
        
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Tesseract OCR extraction failed: {str(e)}")

def get_pdf_text(doc_id):
    """Enhanced PDF text extraction with 2 robust methods"""
    try:
        # Fetch document with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS, timeout=30)
                if response.status_code == 200:
                    break
                elif attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch document after {max_retries} attempts. Status: {response.status_code}")
                else:
                    logger.warning(f"Document fetch attempt {attempt + 1} failed, retrying...", doc_id)
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Network error after {max_retries} attempts: {str(e)}")
                else:
                    logger.warning(f"Network error attempt {attempt + 1}, retrying...", doc_id)
                    time.sleep(2)
        
        # Extract document data
        doc_data = response.json()
        if not doc_data or "value" not in doc_data:
            raise Exception("Invalid response format")
        
        value_data = doc_data["value"]
        daId = value_data["patientId"]["id"] if isinstance(value_data["patientId"], dict) else value_data["patientId"]
        document_buffer = value_data["documentBuffer"]
        
        # Optional metadata
        is_faxed = value_data.get("isFaxed", False)
        fax_source = value_data.get("faxSource", "Unknown")
        document_type = value_data.get("documentType", "Unknown")
        
        # Decode PDF
        pdf_bytes = base64.b64decode(document_buffer)
        if len(pdf_bytes) == 0:
            raise Exception("Empty PDF document buffer")
        
        # Extract text with 2 methods
        text = ""
        extraction_method = "unknown"
        
        # Method 1: PDFPlumber (primary)
        try:
            text = extract_text_pdfplumber(pdf_bytes)
            extraction_method = "pdfplumber"
            if text and text.strip():
                logger.success(f"Text extracted using {extraction_method} ({len(text)} chars)", doc_id)
            else:
                raise Exception("PDFPlumber extracted empty text")
        except Exception as e:
            logger.warning(f"PDFPlumber failed: {str(e)}, trying OCR...", doc_id)
            
            # Method 2: Tesseract OCR (fallback)
            try:
                text = extract_text_tesseract_ocr(pdf_bytes)
                extraction_method = "tesseract_ocr"
                if text and text.strip():
                    logger.success(f"Text extracted using {extraction_method} ({len(text)} chars)", doc_id)
                else:
                    raise Exception("Tesseract OCR extracted empty text")
            except Exception as e:
                logger.error(f"All text extraction methods failed: {str(e)}", doc_id)
                return None
        
        # Clean extracted text
        if text and safe_strip(text):
            # Remove specific patterns and excessive whitespace
            edited = re.sub(r'\b\d[A-Z][A-Z0-9]\d[A-Z][A-Z0-9]\d[A-Z]{2}(?:\d{2})?\b', '', text)
            edited = re.sub(r'\s+', ' ', edited).strip()
        else:
            edited = text
        
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
        return None

# === FIELD NORMALIZATION FUNCTIONS ===
def normalize_sex(sex_value):
    """Normalize sex field to only MALE or FEMALE"""
    if not sex_value:
        return ""
    
    sex_clean = safe_strip(sex_value).upper()
    
    # Handle various formats
    if sex_clean in ['M', 'MALE', 'MAN', 'BOY']:
        return "MALE"
    elif sex_clean in ['F', 'FEMALE', 'WOMAN', 'GIRL']:
        return "FEMALE"
    else:
        return ""

def extract_sex_with_regex(text):
    """Fallback method to extract sex using regex patterns"""
    if not text:
        return ""
    
    # Convert to uppercase for easier matching
    text_upper = text.upper()
    
    # Pattern 1: Sex: M/F or Gender: M/F
    sex_patterns = [
        r'SEX\s*[:=]\s*([MF])\b',
        r'GENDER\s*[:=]\s*([MF])\b',
        r'SEX\s*[:=]\s*(MALE|FEMALE)\b',
        r'GENDER\s*[:=]\s*(MALE|FEMALE)\b',
        r'\bSEX\s+([MF])\b',
        r'\bGENDER\s+([MF])\b',
        r'\b([MF])\s*[/\\]\s*[MF]\b',  # M/F checkbox style
        r'PATIENT.*?SEX\s*[:=]?\s*([MF])\b',
        r'PATIENT.*?GENDER\s*[:=]?\s*([MF])\b',
        r'\[([MF])\]',  # Checkbox style [M] or [F]
        r'\(([MF])\)',  # Parentheses style (M) or (F)
    ]
    
    for pattern in sex_patterns:
        matches = re.findall(pattern, text_upper)
        if matches:
            sex_value = matches[0]
            if sex_value in ['M', 'MALE']:
                return "M"
            elif sex_value in ['F', 'FEMALE']:
                return "F"
    
    return ""

def infer_sex_from_name(first_name):
    """Last resort: infer sex from common first names"""
    if not first_name:
        return ""
    
    first_name_clean = safe_strip(first_name).upper()
    
    # Common male names
    male_names = [
        'MILTON', 'RODERICK', 'ROBERT', 'MICHAEL', 'DAVID', 'JAMES', 'JOHN', 'WILLIAM', 'RICHARD', 'JOSEPH',
        'THOMAS', 'CHRISTOPHER', 'CHARLES', 'DANIEL', 'MATTHEW', 'ANTHONY', 'MARK', 'DONALD', 'STEVEN', 'PAUL',
        'ANDREW', 'JOSHUA', 'KENNETH', 'KEVIN', 'BRIAN', 'GEORGE', 'TIMOTHY', 'RONALD', 'JASON', 'EDWARD',
        'JEFFREY', 'RYAN', 'JACOB', 'GARY', 'NICHOLAS', 'ERIC', 'JONATHAN', 'STEPHEN', 'LARRY', 'JUSTIN',
        'SCOTT', 'BRANDON', 'BENJAMIN', 'SAMUEL', 'FRANK', 'GREGORY', 'RAYMOND', 'ALEXANDER', 'PATRICK', 'JACK'
    ]
    
    # Common female names
    female_names = [
        'LOIS', 'CAROL', 'VIRGINIA', 'MARY', 'PATRICIA', 'JENNIFER', 'LINDA', 'ELIZABETH', 'BARBARA', 'SUSAN',
        'JESSICA', 'SARAH', 'KAREN', 'NANCY', 'LISA', 'BETTY', 'HELEN', 'SANDRA', 'DONNA', 'CAROL', 'RUTH',
        'SHARON', 'MICHELLE', 'LAURA', 'SARAH', 'KIMBERLY', 'DEBORAH', 'DOROTHY', 'AMY', 'ANGELA', 'ASHLEY',
        'BRENDA', 'EMMA', 'OLIVIA', 'CYNTHIA', 'MARIE', 'JANET', 'CATHERINE', 'FRANCES', 'CHRISTINE', 'SAMANTHA',
        'DEBRA', 'RACHEL', 'CAROLYN', 'JANET', 'MARIA', 'HEATHER', 'DIANE', 'JULIE', 'JOYCE', 'VICTORIA'
    ]
    
    if first_name_clean in male_names:
        logger.info(f"Inferred sex as MALE from name: {first_name}")
        return "M"
    elif first_name_clean in female_names:
        logger.info(f"Inferred sex as FEMALE from name: {first_name}")
        return "F"
    
    return ""

def normalize_zip(zip_value):
    """Normalize zip code to handle 4-digit edge case by adding leading zero"""
    if not zip_value:
        return ""
    
    zip_clean = safe_strip(zip_value)
    
    # Remove all non-digit and non-hyphen characters
    zip_clean = re.sub(r'[^\d-]', '', zip_clean)
    
    # Handle 4-digit zip code by adding leading zero
    if zip_clean.isdigit() and len(zip_clean) == 4:
        zip_clean = "0" + zip_clean
        logger.info(f"4-digit zip code detected, added leading zero: {zip_clean}")
    
    return zip_clean

def normalize_mrn(mrn_value):
    """Enhanced MRN normalization to handle various formats"""
    if not mrn_value:
        return ""
    
    mrn_clean = safe_strip(mrn_value)
    
    # Handle different MRN formats
    # Format 1: Pure numbers (2182563)
    if mrn_clean.isdigit() and 4 <= len(mrn_clean) <= 20:
        return mrn_clean
    
    # Format 2: Alphanumeric with letters and numbers (C0200204515501, MA200528053701)
    alphanumeric_clean = re.sub(r'[^a-zA-Z0-9]', '', mrn_clean)
    if 4 <= len(alphanumeric_clean) <= 20:
        return alphanumeric_clean
    
    # Format 3: Numbers with special characters - extract just the numbers if reasonable length
    numbers_only = re.sub(r'[^0-9]', '', mrn_clean)
    if 4 <= len(numbers_only) <= 20:
        return numbers_only
    
    # If none of the above work, return the cleaned alphanumeric version if it exists
    if alphanumeric_clean:
        return alphanumeric_clean
    
    return ""

def extract_patient_data(text):
    """Extract patient data using AI with enhanced field-specific prompts"""
    try:
        if not text or not safe_strip(text):
            logger.warning("Empty text provided for patient data extraction")
            return {}
        
        query = """
        You are an expert in medical documentation. Extract the following fields from the attached medical PDF and return them in the specified JSON format. Use the exact field names below, and provide values based strictly on the document content.
        
        Required JSON format:
        {
        "patientFName": "",
        "patientLName": "",
        "dob": "",
        "patientSex": "",
        "medicalRecordNo": "",
        "billingProvider": "",
        "npi": "",
        "physicianNPI": "",
        "nameOfAgency": "",
        "address": "",
        "city": "",
        "state": "",
        "zip": "",
        "email": "",
        "phoneNumber": "",
        "serviceLine": "",
        "payorSource": "",
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
        
        CRITICAL FIELD EXTRACTION INSTRUCTIONS:
        
        NAMES & DEMOGRAPHICS:
        - patientFName: Extract first name from "Patient Name", "Patient", "Name", "First Name", "Pt Name" fields
        - patientLName: Extract last name from "Patient Name", "Patient", "Name", "Last Name", "Pt Name" fields
        - dob: Extract from "DOB", "Date of Birth", "Birth Date", "Born", "D.O.B.", "Date Of Birth" - format as MM/DD/YYYY
        
        PATIENT SEX/GENDER - CRITICAL PRIORITY:
        - patientSex: This field is EXTREMELY IMPORTANT. Search exhaustively for sex/gender information:
          * PRIMARY LOCATIONS: Patient demographics section, patient information tables, header areas, form fields
          * SEARCH TERMS: "Sex", "Gender", "M/F", "Male/Female", "Sex:", "Gender:", "[M]", "[F]", "(M)", "(F)"
          * FORM ELEMENTS: Look for checkboxes [✓], radio buttons, dropdown selections, filled forms
          * PATTERNS: "Sex: M", "Gender: F", "M/F", checkboxes like "[X] Male [ ] Female"
          * TABLE CELLS: Check patient information tables, demographics grids, header information
          * NEAR PATIENT NAME: Often appears close to patient name or in same section
          * CERTIFICATION FORMS: Check home health certifications, 485 forms, order forms
          * PHYSICIAN ORDERS: Sometimes appears in physician order sections
          * Return EXACTLY "M" or "F" (single letter only)
          * This field should NOT be empty - search the entire document thoroughly
        
        MEDICAL RECORD NUMBER - VERY IMPORTANT:
        - medicalRecordNo: Extract from "MRN", "Medical Record No", "Medical Record Number", "Patient ID", "Chart #", "Record #", "MR#", "MR No", "Medical Record #"
          * Accept ANY format: pure numbers (2182563), alphanumeric (C0200204515501, MA200528053701), mixed formats
          * Look in patient information sections, headers, form fields, ID sections
          * Can be 4-20 characters, numbers only OR letters and numbers combined
          * Examples: "2182563", "C0200204515501", "MA200528053701", "101002392"
          * Do NOT create, generate, or make up any values
        
        SERVICE LINE - VERY IMPORTANT:
        - serviceLine: Extract medical service types from document:
          * Look for: "Physical Therapy", "PT", "Occupational Therapy", "OT", "Speech Therapy", "ST", "SLP"
          * Also look for: "Skilled Nursing", "RN", "LPN", "Nursing", "MSW", "Social Work", "HHA", "Home Health Aide"
          * Medical Social Services: "MSS", "Medical Social Services", "Social Worker"
          * Check service sections, therapy orders, plan of care, skilled services
          * Look in: Order forms, 485 forms, Plan of Care, Service Plans, Therapy Orders
          * Common patterns: "Services Ordered:", "Skilled Services:", "Disciplines:", "Service Types:"
          * Multiple services: Return comma-separated list (e.g., "PT, OT, Nursing")
          * Abbreviations acceptable: "PT", "OT", "ST", "RN", "MSW", "HHA"
        
        PAYER SOURCE - VERY IMPORTANT:
        - payorSource: Extract insurance/payment information:
          * PRIMARY: "Medicare", "Medicaid", "Private Insurance", "Commercial Insurance"
          * SPECIFIC: "Medicare Part A", "Medicare Part B", "Medicare Advantage", "Medicaid"
          * INSURANCE NAMES: "Aetna", "Blue Cross", "Cigna", "Humana", "United Healthcare", "Anthem"
          * PAYMENT TYPES: "Private Pay", "Self Pay", "Workers Comp", "Auto Insurance"
          * LOOK IN: Insurance sections, payer information, billing sections, coverage areas
          * PATTERNS: "Primary Insurance:", "Payer:", "Insurance:", "Coverage:", "Payor Source:"
          * If multiple, prioritize primary insurance or first mentioned
        
        CONTACT INFORMATION:
        - address: Extract full street address from "Address", "Street", "Patient Address", "Home Address", "Mailing Address"
        - city: Extract city from "City", "Patient City", address sections, "Town"
        - state: Extract state from "State", "ST", "Patient State", address sections (2-letter abbreviation preferred)
        - zip: Extract ZIP code from "ZIP", "Zip Code", "Postal Code", address sections (5 or 9 digits)
        - email: Extract from "Email", "E-mail", "Email Address", "Electronic Mail", "E-Mail Address"
        - phoneNumber: Extract from "Phone", "Tel", "Telephone", "Phone Number", "Contact Number", "Home Phone", "Cell Phone" - include area code
        
        PROVIDER INFORMATION:
        - billingProvider: Extract from "Billing Provider", "Provider", "Physician", "Doctor", "Attending", "Primary Care Provider"
        - npi: Extract from "NPI", "Provider NPI", "Physician NPI", "National Provider Identifier" - 10 digits only
        - physicianNPI: Extract from "Physician NPI", "Attending NPI", "Primary Physician NPI", "Primary Care Provider NPI", "Doctor NPI" - 10 digits only
          * Look for physician-specific NPI numbers separate from general provider NPI
          * Check physician signature sections, attending physician areas, primary care provider information
          * May appear as "Dr. [Name] NPI: [number]" or "Physician NPI: [number]"
          * Different from general billing provider NPI - this is specifically for the attending/primary physician
        - nameOfAgency: Extract from "Agency", "Home Health", "Facility", "Organization", "Company", "Agency Name"
        
        EPISODE & DIAGNOSIS - CRITICAL FOR ALL 6 DIAGNOSIS CODES:
        - startOfCare: Look for "SOC", "Start of Care Date", "SOC Date", "Care Start", "Start of Care"
        - startOfEpisode: Look for "Start Date", "Episode Start Date", "From Date", "Episode Begin", "Period From"
        - endOfEpisode: Look for "End Date", "Episode End Date", "To Date", "Episode End", "Period To"
        
        DIAGNOSIS CODES EXTRACTION - EXTREMELY IMPORTANT:
        Extract ALL diagnosis codes from the document. Look for ICD-10-CM format codes in these locations:
        * DIAGNOSIS SECTIONS: "Primary Diagnosis", "Secondary Diagnosis", "Other Diagnoses"
        * PLAN OF CARE: Diagnosis listings, care plan sections
        * 485 FORMS: Home health certification diagnosis areas
        * PHYSICIAN ORDERS: Diagnosis codes in order sections
        * ICD-10 PATTERNS: Letter followed by numbers (e.g., "M25.511", "Z51.11", "I10", "E11.9", "N39.0")
        * DIAGNOSIS TABLES: Numbered diagnosis lists (1. Primary, 2. Secondary, etc.)
        * PATTERNS TO FIND:
          - "Diagnosis 1:", "Primary Dx:", "1st Diagnosis:", "Dx1:"
          - "Diagnosis 2:", "Secondary Dx:", "2nd Diagnosis:", "Dx2:"
          - "Other Diagnosis:", "Additional Diagnosis:", "Comorbidities:"
        * COMMON LOCATIONS: 
          - Plan of Care sections
          - Physician order forms
          - Home health certifications
          - Medical summary sections
          - Diagnosis code tables
        * EXTRACT EXACTLY: Return the exact ICD-10 code (e.g., "M79.3", "Z51.11")
        * PRIORITY ORDER: 
          firstDiagnosis = Primary/Principal diagnosis
          secondDiagnosis = Secondary diagnosis  
          thirdDiagnosis = Third diagnosis
          fourthDiagnosis = Fourth diagnosis
          fifthDiagnosis = Fifth diagnosis
          sixthDiagnosis = Sixth diagnosis
        
        EXTRACTION STRATEGY FOR SEX/GENDER:
        1. Check patient demographics sections FIRST
        2. Look in patient information tables and headers
        3. Search form fields and checkboxes throughout document
        4. Check certification forms (485 forms, home health forms)
        5. Look near patient name and personal information
        6. Check physician order sections and medical forms
        7. Scan for any "M/F", "Male/Female" indicators anywhere in document
        8. Look for filled checkboxes or radio button selections
        
        EXTRACTION STRATEGY FOR DIAGNOSES:
        1. Scan for "ICD-10", "Diagnosis", "Dx" sections first
        2. Look in Plan of Care and 485 certification forms
        3. Check physician order sections for diagnosis codes
        4. Search medical summary and assessment areas
        5. Look for numbered diagnosis lists (Primary, Secondary, etc.)
        6. Check all tables and form fields for diagnosis codes
        7. Extract codes in order of priority/importance
        
        FORMATTING RULES:
        - ALL dates must be in MM/DD/YYYY format (pad with zeros: "4/6/2025" → "04/06/2025")
        - Sex: Return only "M" or "F" (single letter) - MUST be found, search entire document
        - MRN: Return EXACTLY as found - do not modify format, accept numbers, letters, or combinations
        - Service Line: Use standard abbreviations (PT, OT, ST, RN, MSW, HHA) or full names
        - Payer Source: Use standard insurance names (Medicare, Medicaid, etc.)
        - Diagnosis codes: Return exact ICD-10 format (letter + numbers, e.g., "M25.511")
        - Phone numbers: Include area code, format as (XXX) XXX-XXXX if possible
        - ZIP codes: 5 digits or 9 digits with hyphen (XXXXX or XXXXX-XXXX)
        - State: 2-letter abbreviation preferred (CA, NY, TX, etc.)
        - If a field is not found in the document, leave it as empty string ""
        - DO NOT create, generate, or make up any values
        
        Return only the JSON object, no additional text or explanations.
        """
        
        # Try AI extraction with retries
        max_retries = 3  # Increased retries for better accuracy
        for attempt in range(max_retries):
            try:
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
                if not response or not response.text:
                    raise Exception("Empty response from AI model")
                
                # Extract JSON from response
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if not match:
                    raise Exception("No JSON found in AI response")
                
                json_str = match.group()
                extracted_data = json.loads(json_str)
                
                # Validate and normalize required fields
                required_fields = ["patientFName", "patientLName", "dob", "episodeDiagnoses"]
                for field in required_fields:
                    if field not in extracted_data:
                        if field == "episodeDiagnoses":
                            extracted_data[field] = [{"startOfCare": "", "startOfEpisode": "", "endOfEpisode": "", "firstDiagnosis": "", "secondDiagnosis": "", "thirdDiagnosis": "", "fourthDiagnosis": "", "fifthDiagnosis": "", "sixthDiagnosis": ""}]
                        else:
                            extracted_data[field] = ""
                
                # Add and normalize all fields
                all_fields = ["patientSex", "medicalRecordNo", "billingProvider", "npi", "physicianNPI", "nameOfAgency", "address", "city", "state", "zip", "email", "phoneNumber", "serviceLine", "payorSource"]
                for field in all_fields:
                    if field not in extracted_data:
                        extracted_data[field] = ""
                
                # Log what AI extracted for key fields before normalization
                logger.info(f"AI extracted raw sex value: '{extracted_data.get('patientSex', 'NOT_FOUND')}'")
                logger.info(f"AI extracted service line: '{extracted_data.get('serviceLine', 'NOT_FOUND')}'")
                logger.info(f"AI extracted payer source: '{extracted_data.get('payorSource', 'NOT_FOUND')}'")
                
                # Log diagnosis extraction
                episode_data = extracted_data.get("episodeDiagnoses", [{}])[0] if extracted_data.get("episodeDiagnoses") else {}
                logger.info(f"AI extracted diagnoses:")
                for i in range(1, 7):
                    dx_field = f"{'first' if i==1 else 'second' if i==2 else 'third' if i==3 else 'fourth' if i==4 else 'fifth' if i==5 else 'sixth'}Diagnosis"
                    dx_value = episode_data.get(dx_field, "")
                    logger.info(f"  - Diagnosis {i}: '{dx_value}'")
                
                # Apply field normalization
                extracted_data["patientSex"] = normalize_sex(extracted_data.get("patientSex", ""))
                extracted_data["medicalRecordNo"] = normalize_mrn(extracted_data.get("medicalRecordNo", ""))
                
                # If AI didn't find sex, try regex fallback
                if not extracted_data["patientSex"]:
                    regex_sex = extract_sex_with_regex(text)
                    if regex_sex:
                        extracted_data["patientSex"] = normalize_sex(regex_sex)
                    # No logging for fallback attempts - run silently
                
                # Clean and validate other fields
                extracted_data["zip"] = normalize_zip(extracted_data.get("zip", ""))
                extracted_data["phoneNumber"] = re.sub(r'[^\d\(\)\-\s\+]', '', safe_strip(extracted_data.get("phoneNumber", "")))
                extracted_data["npi"] = re.sub(r'[^\d]', '', safe_strip(extracted_data.get("npi", "")))
                extracted_data["physicianNPI"] = re.sub(r'[^\d]', '', safe_strip(extracted_data.get("physicianNPI", "")))
                
                # Clean service line and payer source
                extracted_data["serviceLine"] = safe_strip(extracted_data.get("serviceLine", ""))
                extracted_data["payorSource"] = safe_strip(extracted_data.get("payorSource", ""))
                
                # Log final results for debugging
                logger.info(f"Final extracted MRN: '{extracted_data.get('medicalRecordNo', 'NOT_FOUND')}'")
                logger.info(f"Final extracted Sex: '{extracted_data.get('patientSex', 'NOT_FOUND')}'")
                logger.info(f"Final extracted Service Line: '{extracted_data.get('serviceLine', 'NOT_FOUND')}'")
                logger.info(f"Final extracted Payer Source: '{extracted_data.get('payorSource', 'NOT_FOUND')}'")
                
                logger.success("Patient data extraction completed successfully")
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All JSON parsing attempts failed")
                else:
                    time.sleep(1)
                    
            except Exception as e:
                logger.warning(f"AI extraction attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All AI extraction attempts failed")
                else:
                    time.sleep(1)
        
        # Return minimal structure if extraction fails
        logger.warning("AI extraction failed, returning minimal patient data structure")
        return {
            "patientFName": "",
            "patientLName": "",
            "dob": "",
            "patientSex": "",
            "medicalRecordNo": "",
            "billingProvider": "",
            "npi": "",
            "physicianNPI": "",
            "nameOfAgency": "",
            "address": "",
            "city": "",
            "state": "",
            "zip": "",
            "email": "",
            "phoneNumber": "",
            "serviceLine": "",
            "payorSource": "",
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
        return {}

def extract_order_data(text):
    """Extract order data using AI with enhanced field-specific prompts"""
    try:
        if not text or not safe_strip(text):
            logger.warning("Empty text provided for order data extraction")
            return {}
        
        query = """
        You are an expert in medical documentation. Extract the following fields from the attached medical PDF. Return the data in the following JSON format:

        {
        "orderNo": "",
        "orderDate": "",
        "startOfCare": "",
        "episodeStartDate": "",
        "episodeEndDate": "",
        "documentID": "",
        "mrn": "",
        "patientName": "",
        "patientSex": "",
        "patientAddress": "",
        "patientCity": "",
        "patientState": "",
        "patientZip": "",
        "patientPhone": "",
        "serviceLine": "",
        "payorSource": "",
        "sentToPhysicianDate": "",
        "sentToPhysicianStatus": false,
        "patientId": "",
        "companyId": "",
        "bit64Url": "",
        "documentName": ""
        }

        CRITICAL FIELD EXTRACTION INSTRUCTIONS:
        
        ORDER INFORMATION:
        - orderNo: Look for "Order #", "Order No", "Order Number", "Reference #", "Order ID", "Form #", "Document #"
        - orderDate: Look for "Order Date", "Date Ordered", "Created Date", "Form Date", "Date of Service"
        - documentName: Look for document type like "Certification", "Recertification", "Orders", "485", "Plan of Care", "Home Health Certification"
        
        EPISODE DATES:
        - startOfCare: Look for "SOC", "Start of Care", "SOC Date", "Care Start Date", "Start of Care Date"
        - episodeStartDate: Look for "Start Date", "Episode Start", "From Date", "Period From", "Service From", "Episode Start Date"
        - episodeEndDate: Look for "End Date", "Episode End", "To Date", "Period To", "Service To", "Episode End Date"
        
        PATIENT IDENTIFICATION - VERY IMPORTANT:
        - mrn: Look for "MRN", "Medical Record No", "Medical Record Number", "Patient ID", "Chart #", "Record #", "MR#", "MR No", "Medical Record #", "Patient MRN"
          * Accept ANY format: pure numbers (2182563), alphanumeric (C0200204515501, MA200528053701), mixed formats
          * Look in patient information sections, headers, form fields, ID sections, order headers
          * Can be 4-20 characters, numbers only OR letters and numbers combined
          * Examples: "2182563", "C0200204515501", "MA200528053701", "101002392"
          * Search in order forms, patient identification areas, document headers
          * Do NOT create, generate, or make up any values
        - patientName: Look for "Patient Name", "Patient", "Name", "Pt Name", full patient name in order forms
        - patientSex: Extract patient sex/gender from order documents:
          * Look for "Sex", "Gender", "M/F", "Male/Female", "Sex:", "Gender:" in patient sections
          * Check patient demographics in order forms, certification documents
          * Search form fields, checkboxes, dropdown selections
          * Return EXACTLY "M" or "F" (single letter only)
        
        SERVICE LINE INFORMATION - VERY IMPORTANT:
        - serviceLine: Extract medical service types from order documents:
          * Look for: "Physical Therapy", "PT", "Occupational Therapy", "OT", "Speech Therapy", "ST", "SLP"
          * Also look for: "Skilled Nursing", "RN", "LPN", "Nursing", "MSW", "Social Work", "HHA", "Home Health Aide"
          * Medical Social Services: "MSS", "Medical Social Services", "Social Worker"
          * Check service sections, therapy orders, plan of care, skilled services, disciplines ordered
          * Look in: Order forms, 485 forms, Plan of Care, Service Plans, Therapy Orders, Service Authorizations
          * Common patterns: "Services Ordered:", "Skilled Services:", "Disciplines:", "Service Types:", "Therapy Services:"
          * ORDER SECTIONS: "Physician Orders", "Services to be Provided", "Skilled Services", "Disciplines"
          * Multiple services: Return comma-separated list (e.g., "PT, OT, Nursing")
          * Abbreviations acceptable: "PT", "OT", "ST", "RN", "MSW", "HHA"
        
        PAYER SOURCE INFORMATION - VERY IMPORTANT:
        - payorSource: Extract insurance/payment information from order documents:
          * PRIMARY: "Medicare", "Medicaid", "Private Insurance", "Commercial Insurance"
          * SPECIFIC: "Medicare Part A", "Medicare Part B", "Medicare Advantage", "Medicaid"
          * INSURANCE NAMES: "Aetna", "Blue Cross", "Cigna", "Humana", "United Healthcare", "Anthem"
          * PAYMENT TYPES: "Private Pay", "Self Pay", "Workers Comp", "Auto Insurance"
          * LOOK IN: Insurance sections, payer information, billing sections, coverage areas, certification forms
          * PATTERNS: "Primary Insurance:", "Payer:", "Insurance:", "Coverage:", "Payor Source:", "Payment Source:"
          * CERTIFICATION FORMS: Look in 485 forms for insurance information
          * If multiple, prioritize primary insurance or first mentioned
        
        PATIENT CONTACT (if available):
        - patientAddress: Look for "Patient Address", "Address", "Street Address", "Home Address", "Mailing Address"
        - patientCity: Look for "City", "Patient City", "Town"
        - patientState: Look for "State", "ST", "Patient State" 
        - patientZip: Look for "ZIP", "Zip Code", "Postal Code"
        - patientPhone: Look for "Phone", "Tel", "Telephone", "Phone Number", "Contact Number", "Home Phone", "Cell Phone"
        
        ADMINISTRATIVE:
        - sentToPhysicianDate: Look for "Sent to Physician", "Physician Date", "Forwarded Date"
        - sentToPhysicianStatus: Look for physician signature status - return true/false
        
        EXTRACTION STRATEGY:
        1. Scan the ENTIRE document including headers, footers, tables, forms
        2. Look for order forms, certification documents, patient information sections
        3. Check form fields, patient identification areas, document headers
        4. Search in order summary sections, patient demographics, ID fields
        5. Look for MRN/ID numbers prominently displayed in headers or patient info
        6. Check order forms, medical record sections, patient identification areas
        7. PRIORITIZE service and payer information in order sections, 485 forms, and plan of care
        8. Look for therapy orders, skilled service authorizations, insurance certifications
        
        FORMATTING RULES:
        - ALL dates must be in MM/DD/YYYY format (pad with zeros: "4/6/2025" → "04/06/2025")
        - MRN: Return EXACTLY as found - do not modify format, accept numbers, letters, or combinations
        - Service Line: Use standard abbreviations (PT, OT, ST, RN, MSW, HHA) or full names
        - Payer Source: Use standard insurance names (Medicare, Medicaid, etc.)
        - Phone numbers: Include area code if available
        - ZIP codes: 5 digits or 9 digits with hyphen
        - State: 2-letter abbreviation preferred
        - sentToPhysicianStatus: must be boolean (true/false)
        - Return missing fields as "" (empty string) or false for booleans
        - Extract from tables, forms, headers, and all text sections
        - DO NOT create, generate, or make up any values
        
        Return only the JSON object, no additional text or explanations.
        """
        
        # Try AI extraction
        max_retries = 3  # Increased retries for better accuracy
        for attempt in range(max_retries):
            try:
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
                if not response or not response.text:
                    raise Exception("Empty response from AI model")
                
                # Extract JSON from response
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if not match:
                    raise Exception("No JSON found in AI response")
                
                json_str = match.group()
                extracted_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["orderNo", "orderDate", "startOfCare", "episodeStartDate", "episodeEndDate", "documentID", "mrn", "patientName", "sentToPhysicianDate", "sentToPhysicianStatus", "patientId", "companyId", "bit64Url", "documentName"]
                for field in required_fields:
                    if field not in extracted_data:
                        if field == "sentToPhysicianStatus":
                            extracted_data[field] = False
                        else:
                            extracted_data[field] = ""
                
                # Add new fields if missing (including service line and payer source)
                additional_fields = ["patientAddress", "patientCity", "patientState", "patientZip", "patientPhone", "serviceLine", "payorSource"]
                for field in additional_fields:
                    if field not in extracted_data:
                        extracted_data[field] = ""
                
                # Normalize sex and MRN in order data too
                extracted_data["patientSex"] = normalize_sex(extracted_data.get("patientSex", ""))
                extracted_data["mrn"] = normalize_mrn(extracted_data.get("mrn", ""))
                
                # Clean other fields
                extracted_data["patientZip"] = normalize_zip(extracted_data.get("patientZip", ""))
                extracted_data["patientPhone"] = re.sub(r'[^\d\(\)\-\s\+]', '', safe_strip(extracted_data.get("patientPhone", "")))
                
                # Clean service line and payer source
                extracted_data["serviceLine"] = safe_strip(extracted_data.get("serviceLine", ""))
                extracted_data["payorSource"] = safe_strip(extracted_data.get("payorSource", ""))
                
                # Log what was found for debugging
                logger.info(f"Order extraction - MRN: '{extracted_data.get('mrn', 'NOT_FOUND')}'")
                logger.info(f"Order extraction - Sex: '{extracted_data.get('patientSex', 'NOT_FOUND')}'")
                logger.info(f"Order extraction - Service Line: '{extracted_data.get('serviceLine', 'NOT_FOUND')}'")
                logger.info(f"Order extraction - Payer Source: '{extracted_data.get('payorSource', 'NOT_FOUND')}'")
                
                logger.success("Order data extraction completed successfully")
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All JSON parsing attempts failed")
                else:
                    time.sleep(1)
                    
            except Exception as e:
                logger.warning(f"AI extraction attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All AI extraction attempts failed")
                else:
                    time.sleep(1)
        
        # Return minimal structure if extraction fails
        return {
            "orderNo": "",
            "orderDate": "",
            "startOfCare": "",
            "episodeStartDate": "",
            "episodeEndDate": "",
            "documentID": "",
            "mrn": "",
            "patientName": "",
            "patientSex": "",
            "patientAddress": "",
            "patientCity": "",
            "patientState": "",
            "patientZip": "",
            "patientPhone": "",
            "serviceLine": "",
            "payorSource": "",
            "sentToPhysicianDate": "",
            "sentToPhysicianStatus": False,
            "patientId": "",
            "companyId": "",
            "bit64Url": "",
            "documentName": ""
        }
        
    except Exception as e:
        logger.error(f"Critical error in extract_order_data: {str(e)}")
        return {}

def fetch_signed_date(doc_id):
    """Fetch signed date from document status API"""
    try:
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
    except Exception as e:
        logger.warning(f"Failed to fetch signed date: {str(e)}", doc_id)
        return None

def get_patient_details_from_api(patient_id):
    """Get patient details from API"""
    try:
        url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/get-patient/{patient_id}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to fetch patient details for patient ID: {patient_id}")
            return {}
    except Exception as e:
        logger.warning(f"Error fetching patient details: {str(e)}")
        return {}

def create_patient_via_api(patient_data, doc_id):
    """Create patient via API and return patient ID"""
    try:
        # Get episode data
        episode_info = patient_data.get("episodeDiagnoses", [{}])[0] if patient_data.get("episodeDiagnoses") else {}
        
        # Prepare patient creation payload
        payload = {
            "agencyInfo": {
                "patientFName": patient_data.get("patientFName", ""),
                "patientLName": patient_data.get("patientLName", ""),
                "dob": patient_data.get("dob", ""),
                "patientSex": patient_data.get("patientSex", ""),
                "medicalRecordNo": patient_data.get("medicalRecordNo", ""),
                "billingProvider": patient_data.get("billingProvider", ""),
                "npi": patient_data.get("npi", ""),
                "physicianNPI": patient_data.get("physicianNPI", ""),
                "nameOfAgency": patient_data.get("nameOfAgency", ""),
                "address": patient_data.get("address", ""),
                "city": patient_data.get("city", ""),
                "state": patient_data.get("state", ""),
                "zip": patient_data.get("zip", ""),
                "email": patient_data.get("email", ""),
                "phoneNumber": patient_data.get("phoneNumber", ""),
                "serviceLine": patient_data.get("serviceLine", ""),
                "payorSource": patient_data.get("payorSource", "")
            },
            "episodeDiagnoses": [
                {
                    "startOfCare": episode_info.get("startOfCare", ""),
                    "startOfEpisode": episode_info.get("startOfEpisode", ""),
                    "endOfEpisode": episode_info.get("endOfEpisode", ""),
                    "firstDiagnosis": episode_info.get("firstDiagnosis", ""),
                    "secondDiagnosis": episode_info.get("secondDiagnosis", ""),
                    "thirdDiagnosis": episode_info.get("thirdDiagnosis", ""),
                    "fourthDiagnosis": episode_info.get("fourthDiagnosis", ""),
                    "fifthDiagnosis": episode_info.get("fifthDiagnosis", ""),
                    "sixthDiagnosis": episode_info.get("sixthDiagnosis", "")
                }
            ]
        }
        
        logger.info(f"Creating patient via API: {patient_data.get('patientFName')} {patient_data.get('patientLName')}", doc_id)
        
        # Make API call
        response = requests.post(PATIENT_CREATE_URL, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            response_data = response.json()
            patient_id = response_data.get("patientId")
            
            if patient_id:
                logger.success(f"Patient created successfully with ID: {patient_id}", doc_id)
                
                # Save to local cache
                patient_key = f"{patient_data.get('patientFName', '').lower()}_{patient_data.get('patientLName', '').lower()}_{patient_data.get('dob', '')}"
                created_patients[patient_key] = {
                    "patient_id": patient_id,
                    "first_name": patient_data.get("patientFName", ""),
                    "last_name": patient_data.get("patientLName", ""),
                    "dob": patient_data.get("dob", ""),
                    "created_at": datetime.now().isoformat(),
                    "doc_id": doc_id
                }
                
                # Save updated cache
                with open("hawthorn_internalmedicine.json", "w") as f:
                    json.dump(created_patients, f, indent=2)
                
                return patient_id
            else:
                logger.error(f"Patient creation response missing patient ID", doc_id)
                return None
        else:
            logger.error(f"Patient creation failed with status {response.status_code}: {response.text}", doc_id)
            return None
            
    except Exception as e:
        logger.error(f"Error creating patient via API: {str(e)}", doc_id)
        return None

def push_order_via_api(order_data, patient_id, doc_id):
    """Push order data via API"""
    try:
        # Prepare order payload
        payload = {
            "orderNo": order_data.get("orderNo", ""),
            "orderDate": order_data.get("orderDate", ""),
            "startOfCare": order_data.get("startOfCare", ""),
            "episodeStartDate": order_data.get("episodeStartDate", ""),
            "episodeEndDate": order_data.get("episodeEndDate", ""),
            "documentID": order_data.get("documentID", ""),
            "mrn": order_data.get("mrn", ""),
            "patientName": order_data.get("patientName", ""),
            "sentToPhysicianDate": order_data.get("sentToPhysicianDate", ""),
            "sentToPhysicianStatus": order_data.get("sentToPhysicianStatus", False),
            "signedByPhysicianDate": order_data.get("signedByPhysicianDate", ""),
            "signedByPhysicianStatus": bool(order_data.get("signedByPhysicianDate")),
            "uploadedSignedOrderStatus": bool(order_data.get("signedByPhysicianDate")),
            "patientId": patient_id,
            "companyId": order_data.get("companyId", ""),
            "pgCompanyId": order_data.get("pgCompanyId", ""),
            "bit64Url": order_data.get("bit64Url", ""),
            "documentName": order_data.get("documentName", "")
        }
        
        logger.info(f"Pushing order via API for patient ID: {patient_id}", doc_id)
        
        # Make API call
        response = requests.post(ORDER_PUSH_URL, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            logger.success(f"Order pushed successfully for patient ID: {patient_id}", doc_id)
            return {"success": True, "status_code": 200, "response": response.text}
        else:
            logger.error(f"Order push failed with status {response.status_code}: {response.text}", doc_id)
            return {"success": False, "status_code": response.status_code, "response": response.text}
            
    except Exception as e:
        logger.error(f"Error pushing order via API: {str(e)}", doc_id)
        return {"success": False, "status_code": None, "response": str(e)}

def check_patient_exists_comprehensive(patient_data, doc_id):
    """Check if patient exists in local cache or via API"""
    try:
        fname = patient_data.get("patientFName", "").strip()
        lname = patient_data.get("patientLName", "").strip()
        dob = patient_data.get("dob", "").strip()
        
        if not fname or not lname or not dob:
            logger.warning(f"Insufficient patient data for lookup: {fname} {lname} {dob}", doc_id)
            return None

        # Check local cache first
        patient_key = f"{fname.lower()}_{lname.lower()}_{dob}"
        if patient_key in created_patients:
            patient_id = created_patients[patient_key]["patient_id"]
            logger.success(f"Patient found in local cache: {fname} {lname}, ID: {patient_id}", doc_id)
            return patient_id
        
        # Check via API (similar to main.py logic)
        check_ids = [PG_ID]
        fname_upper = fname.upper()
        lname_upper = lname.upper()
        
        for check_id in check_ids:
            try:
                url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/company/pg/{check_id}"
                response = requests.get(url, headers=HEADERS)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch patient list for company ID: {check_id}", doc_id)
                    continue
                
                patients = response.json()
                for p in patients:
                    info = p.get("agencyInfo", {})
                    if not info:
                        continue
                    
                    existing_fname = (info.get("patientFName") or "").strip().upper()
                    existing_lname = (info.get("patientLName") or "").strip().upper()
                    existing_dob = info.get("dob", "").strip()
                    
                    if existing_fname == fname_upper and existing_lname == lname_upper and existing_dob == dob:
                        patient_id = p.get("id") or p.get("patientId")
                        logger.success(f"Patient exists in API: {fname} {lname}, ID: {patient_id}", doc_id)
                        
                        # Save to local cache for future lookups
                        created_patients[patient_key] = {
                            "patient_id": patient_id,
                            "first_name": fname,
                            "last_name": lname,
                            "dob": dob,
                            "found_via_api": True,
                            "found_at": datetime.now().isoformat()
                        }
                        
                        with open("hawthorn_internalmedicine.json", "w") as f:
                            json.dump(created_patients, f, indent=2)
                        
                        return patient_id
                        
            except Exception as e:
                logger.warning(f"Error checking patients via API: {str(e)}", doc_id)
                continue
        
        logger.info(f"Patient not found in cache or API: {fname} {lname}", doc_id)
        return None
        
    except Exception as e:
        logger.error(f"Error in comprehensive patient check: {str(e)}", doc_id)
        return None

def save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, csv_writer):
    """Save detailed information about what was pushed to API"""
    try:
        # Get episode data
        episode_info = patient_data.get("episodeDiagnoses", [{}])[0] if patient_data.get("episodeDiagnoses") else {}
        
        # Generate detailed remarks based on what happened
        remarks = []
        
        # Check processing status
        if api_results.get('status') == 'SUCCESS':
            remarks.append("✅ Successfully processed")
            if api_results.get('patient_created'):
                remarks.append("New patient created")
            else:
                remarks.append("Existing patient found")
            if api_results.get('order_pushed'):
                remarks.append("Order pushed to API")
        else:
            remarks.append("❌ Processing failed")
            
        # Add specific error details
        error_msg = api_results.get('error_message', '')
        if error_msg:
            if "Could not extract text from PDF" in error_msg:
                remarks.append("PDF text extraction failed")
            elif "Insufficient patient data" in error_msg:
                remarks.append("Missing required patient information")
            elif "Invalid patient name blocked" in error_msg:
                remarks.append("🚫 INVALID PATIENT NAME BLOCKED - Production safety measure")
            elif "Insufficient date information" in error_msg:
                remarks.append("Missing or invalid dates")
            elif "Missing company ID" in error_msg:
                remarks.append("Agency not found in company mapping")
            elif "Missing episode start date" in error_msg:
                remarks.append("Episode start date missing")
            elif "Missing episode end date" in error_msg:
                remarks.append("Episode end date missing")
            elif "Missing start of care" in error_msg:
                remarks.append("Start of care date missing")
            elif "Failed to create patient" in error_msg:
                remarks.append("API patient creation failed")
            elif "Failed to push order" in error_msg:
                remarks.append("API order push failed")
                # Check if it's a duplicate order
                if api_results.get('duplicate_order_id'):
                    remarks.append(f"🔄 Duplicate order detected (ID: {api_results.get('duplicate_order_id')})")
            else:
                remarks.append(f"Error: {error_msg}")
        
        # Add data quality remarks (informational only - not errors)
        if patient_data:
            if not patient_data.get("medicalRecordNo"):
                remarks.append("⚠️ Medical record number missing")
            if not patient_data.get("serviceLine"):
                remarks.append("⚠️ Service line not identified")
            if not patient_data.get("payorSource"):
                remarks.append("⚠️ Payer source not found")
            # Note: Removed patient sex as an error condition - it's now just informational
        
        # Join all remarks
        final_remarks = " | ".join(remarks) if remarks else "No specific remarks"
        
        # Create detailed row for API push tracking
        api_row = {
            'Document_ID': doc_id,
            'Timestamp': datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            'Patient_ID': patient_id or 'FAILED',
            'Patient_Created': api_results.get('patient_created', False),
            'Order_Pushed': api_results.get('order_pushed', False),
            'Patient_First_Name': patient_data.get("patientFName", ""),
            'Patient_Last_Name': patient_data.get("patientLName", ""),
            'Patient_DOB': patient_data.get("dob", ""),
            'Patient_Sex': patient_data.get("patientSex", ""),
            'Medical_Record_No': patient_data.get("medicalRecordNo", ""),
            'Service_Line': patient_data.get("serviceLine", ""),
            'Payer_Source': patient_data.get("payorSource", ""),
            'Physician_NPI': patient_data.get("physicianNPI", ""),
            'Agency_Name': patient_data.get("nameOfAgency", ""),
            'Patient_Address': patient_data.get("address", ""),
            'Patient_City': patient_data.get("city", ""),
            'Patient_State': patient_data.get("state", ""),
            'Patient_Zip': patient_data.get("zip", ""),
            'Patient_Phone': patient_data.get("phoneNumber", ""),
            'Patient_Email': patient_data.get("email", ""),
            'Order_Number': order_data.get("orderNo", ""),
            'Order_Date': order_data.get("orderDate", ""),
            'Start_Of_Care': order_data.get("startOfCare", ""),
            'Episode_Start_Date': order_data.get("episodeStartDate", ""),
            'Episode_End_Date': order_data.get("episodeEndDate", ""),
            'Sent_To_Physician_Date': order_data.get("sentToPhysicianDate", ""),
            'Signed_By_Physician_Date': order_data.get("signedByPhysicianDate", ""),
            'Company_ID': order_data.get("companyId", ""),
            'PG_Company_ID': order_data.get("pgCompanyId", ""),
            'SOC_Episode': episode_info.get("startOfCare", ""),
            'Start_Episode': episode_info.get("startOfEpisode", ""),
            'End_Episode': episode_info.get("endOfEpisode", ""),
            'Diagnosis_1': episode_info.get("firstDiagnosis", ""),
            'Diagnosis_2': episode_info.get("secondDiagnosis", ""),
            'Diagnosis_3': episode_info.get("thirdDiagnosis", ""),
            'Diagnosis_4': episode_info.get("fourthDiagnosis", ""),
            'Diagnosis_5': episode_info.get("fifthDiagnosis", ""),
            'Diagnosis_6': episode_info.get("sixthDiagnosis", ""),
            'API_Status': api_results.get('status', 'UNKNOWN'),
            'Error_Message': api_results.get('error_message', ''),
            'Remarks': final_remarks
        }
        
        # Write to CSV
        csv_writer.writerow(api_row)
        logger.info(f"API push details saved for document {doc_id}", doc_id)
        return True
        
    except Exception as e:
        logger.error(f"Error saving API push details: {str(e)}", doc_id)
        return False

def process_dates_for_patient(patient_data, doc_id, audit_reason=None):
    """Process and validate dates for patient data"""
    try:
        # Normalize date of birth
        dob = patient_data.get("dob", "")
        if dob:
            try:
                # Parse and reformat DOB to ensure MM/DD/YYYY format
                dob_dt = datetime.strptime(dob, "%m/%d/%Y")
                patient_data["dob"] = dob_dt.strftime("%m/%d/%Y")
                logger.info(f"DOB normalized: {patient_data['dob']}", doc_id)
            except ValueError:
                logger.warning(f"Invalid DOB format: {dob}", doc_id)
                if audit_reason:
                    with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, f"Invalid DOB format: {dob}"])
        
        # Process episode dates if available
        episode_data = patient_data.get("episodeDiagnoses", [])
        if episode_data and len(episode_data) > 0:
            episode = episode_data[0]
            
            # Normalize episode dates
            for date_field in ["startOfCare", "startOfEpisode", "endOfEpisode"]:
                date_value = episode.get(date_field, "")
                if date_value:
                    try:
                        date_dt = datetime.strptime(date_value, "%m/%d/%Y")
                        episode[date_field] = date_dt.strftime("%m/%d/%Y")
                        logger.info(f"Episode {date_field} normalized: {episode[date_field]}", doc_id)
                    except ValueError:
                        logger.warning(f"Invalid {date_field} format: {date_value}", doc_id)
        
        # Validate required fields
        if not patient_data.get("patientFName") or not patient_data.get("patientLName"):
            logger.warning(f"Missing required patient name fields", doc_id)
            if audit_reason:
                with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Missing required patient name fields"])
            return None
        
        if not patient_data.get("dob"):
            logger.warning(f"Missing required DOB field", doc_id)
            if audit_reason:
                with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Missing required DOB field"])
            return None
        
        logger.success(f"Patient date processing completed successfully", doc_id)
        return patient_data
        
    except Exception as e:
        logger.error(f"Error processing patient dates: {str(e)}", doc_id)
        if audit_reason:
            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([doc_id, f"Date processing error: {str(e)}"])
        return None

def process_csv(csv_path):
    """Main CSV processing function - extracts data, pushes to API, and saves to output CSV"""
    logger.info(f"Starting CSV processing with API integration: {csv_path}")
    
    processed_count = 0
    success_count = 0
    error_count = 0
    api_success_count = 0
    api_error_count = 0
    
    # Output CSV files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"csv_outputs/extracted_patients_{timestamp}.csv"
    api_details_filename = f"api_outputs/api_push_details_{timestamp}.csv"
    
    # CSV headers matching the expected format
    csv_headers = [
        'ID', 'Created At', 'Created By', 'Is Billable', 'Is PG Billable', 'Is Eligible', 'Is PG Eligible',
        'Patient WAV ID', 'Patient EHR Record ID', 'Patient EHR Type', 'First Name', 'Middle Name', 'Last Name',
        'Profile Picture', 'Date of Birth', 'Age', 'Sex', 'Status', 'Marital Status', 'SSN', 'Start of Care',
        'Medical Record No', 'Service Line', 'Address', 'City', 'State', 'Zip', 'Email', 'Phone Number', 'Fax',
        'Payor Source', 'Billing Provider', 'Billing Provider Phone', 'Billing Provider Address', 'Billing Provider Zip',
        'NPI', 'Line1 DOS From', 'Line1 DOS To', 'Line1 POS', 'Physician NPI', 'Supervising Provider',
        'Supervising Provider NPI', 'Physician Group', 'Physician Group NPI', 'Physician Group Address',
        'Physician Phone', 'Physician Address', 'City State Zip', 'Patient Account No', 'Agency NPI',
        'Name of Agency', 'Insurance ID', 'Primary Insurance', 'Secondary Insurance', 'Secondary Insurance ID',
        'Tertiary Insurance', 'Tertiary Insurance ID', 'Next of Kin', 'Patient Caretaker', 'Caretaker Contact Number',
        'Remarks', 'DA Backoffice ID', 'Company ID', 'PG Company ID', 'Signed Orders', 'Total Orders',
        'Latest_Episode_StartOfCare', 'Latest_Episode_StartOfEpisode', 'Latest_Episode_EndOfEpisode',
        'Latest_Episode_FirstDiagnosis', 'Latest_Episode_SecondDiagnosis', 'Latest_Episode_ThirdDiagnosis',
        'Latest_Episode_FourthDiagnosis', 'Latest_Episode_FifthDiagnosis', 'Latest_Episode_SixthDiagnosis'
    ]
    
    # API details CSV headers
    api_headers = [
        'Document_ID', 'Timestamp', 'Patient_ID', 'Patient_Created', 'Order_Pushed', 'Patient_First_Name',
        'Patient_Last_Name', 'Patient_DOB', 'Patient_Sex', 'Medical_Record_No', 'Service_Line', 'Payer_Source',
        'Physician_NPI', 'Agency_Name', 'Patient_Address', 'Patient_City', 'Patient_State', 'Patient_Zip',
        'Patient_Phone', 'Patient_Email', 'Order_Number', 'Order_Date', 'Start_Of_Care', 'Episode_Start_Date',
        'Episode_End_Date', 'Sent_To_Physician_Date', 'Signed_By_Physician_Date', 'Company_ID', 'PG_Company_ID',
        'SOC_Episode', 'Start_Episode', 'End_Episode', 'Diagnosis_1', 'Diagnosis_2', 'Diagnosis_3',
        'Diagnosis_4', 'Diagnosis_5', 'Diagnosis_6', 'API_Status', 'Error_Message', 'Remarks'
    ]
    
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as output_file, \
             open(api_details_filename, 'w', newline='', encoding='utf-8') as api_file:
            
            csv_writer = csv.DictWriter(output_file, fieldnames=csv_headers)
            csv_writer.writeheader()
            
            api_writer = csv.DictWriter(api_file, fieldnames=api_headers)
            api_writer.writeheader()
            
            logger.success(f"Created output CSV file: {output_filename}")
            logger.success(f"Created API details CSV file: {api_details_filename}")
            
            # Row counter logic: only process first 10 documents (i = 0,1,2,...,9)
            i = 0
            with open(csv_path, newline='', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)
                
                for row in reader:
                    # Limit processing: only process first 10 documents
                    if i >= 5:
                        break
                    i += 1
                    processed_count += 1
                    
                    doc_id = row["ID"]
                    agency = safe_strip(row.get("Facility"))
                    received = normalize_date_string(safe_strip(row.get("Received On")))
                    
                    logger.header(f"Processing Document ID: {doc_id}")
                    logger.info(f"Agency: {agency}")
                    logger.info(f"Date: {received}")
                    
                    api_results = {
                        'patient_created': False,
                        'order_pushed': False,
                        'status': 'FAILED',
                        'error_message': ''
                    }
                    patient_id = None
                    
                    try:
                        # Step 1: Extract text from PDF
                        logger.progress("\n🔄 Step 1: Extracting text from PDF...")
                        res = get_pdf_text(doc_id)
                        if not res:
                            logger.error(f"Failed to extract text for Doc ID {doc_id}")
                            api_results['error_message'] = "Could not extract text from PDF"
                            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Could not extract text from PDF"])
                            error_count += 1
                            # Save API details for failed extraction
                            save_api_push_details(doc_id, {}, {}, None, api_results, api_writer)
                            continue
                        
                        text, daId, metadata = res
                        
                        if not text or not text.strip():
                            raise ValueError("Empty text extracted from PDF")
                        
                        logger.success(f"   ✅ Text extracted using {metadata['extractionMethod']} ({len(text)} chars)")
                        
                        # Step 2: Extract patient data
                        logger.progress("\n🔄 Step 2: Extracting patient data...")
                        patient_data = extract_patient_data(text)
                        logger.success(f"   ✅ Patient data extracted successfully")
                        logger.data("Patient Response", patient_data)
                        
                        # Clean the data using remove_none_fields
                        patient_data = remove_none_fields(patient_data)
                        
                        if not patient_data or not patient_data.get("patientFName") or not patient_data.get("dob"):
                            logger.warning(f"   ❌ Insufficient patient data extracted")
                            api_results['error_message'] = "Insufficient patient data extracted"
                            error_count += 1
                            # Save API details for insufficient data
                            save_api_push_details(doc_id, patient_data, {}, None, api_results, api_writer)
                            continue
                        
                        # ENHANCED VALIDATION: Check for invalid patient names
                        first_name = safe_strip(patient_data.get("patientFName", ""))
                        last_name = safe_strip(patient_data.get("patientLName", ""))
                        
                        # Check for invalid name patterns
                        invalid_name_patterns = [
                            r'^Unknown_\d+$',  # Unknown_9311341
                            r'^DOC_?\d+$',     # DOC9311341 or DOC_9311341
                            r'^Document_?\d+$', # Document9311341
                            r'^ID_?\d+$',      # ID9311341
                            r'^\d+$',          # Pure numbers like 9311341
                            r'^[A-Z]+_\d+$',   # Any pattern like ABC_123456
                            r'^Patient_?\d+$', # Patient9311341
                            r'^Record_?\d+$',  # Record9311341
                        ]
                        
                        name_validation_failed = False
                        validation_reason = ""
                        
                        # Check first name
                        for pattern in invalid_name_patterns:
                            if re.match(pattern, first_name, re.IGNORECASE):
                                name_validation_failed = True
                                validation_reason = f"Invalid first name pattern: '{first_name}'"
                                break
                        
                        # Check last name if first name passed
                        if not name_validation_failed:
                            for pattern in invalid_name_patterns:
                                if re.match(pattern, last_name, re.IGNORECASE):
                                    name_validation_failed = True
                                    validation_reason = f"Invalid last name pattern: '{last_name}'"
                                    break
                        
                        # Additional checks for suspicious names
                        if not name_validation_failed:
                            # Check if name contains only numbers or is too short
                            if len(first_name) < 2 or len(last_name) < 2:
                                name_validation_failed = True
                                validation_reason = f"Names too short: '{first_name}' '{last_name}'"
                            elif first_name.isdigit() or last_name.isdigit():
                                name_validation_failed = True
                                validation_reason = f"Names contain only numbers: '{first_name}' '{last_name}'"
                            elif "unknown" in first_name.lower() or "unknown" in last_name.lower():
                                name_validation_failed = True
                                validation_reason = f"Names contain 'unknown': '{first_name}' '{last_name}'"
                        
                        # If name validation failed, skip this document
                        if name_validation_failed:
                            logger.error(f"   ❌ INVALID PATIENT NAME DETECTED: {validation_reason}")
                            logger.error(f"   🚫 BLOCKING API PUSH to prevent invalid data in production")
                            api_results['error_message'] = f"Invalid patient name blocked: {validation_reason}"
                            error_count += 1
                            # Save API details for invalid name
                            save_api_push_details(doc_id, patient_data, {}, None, api_results, api_writer)
                            continue
                        
                        logger.success(f"   ✅ Patient identified: {patient_data.get('patientFName')} {patient_data.get('patientLName')}")
                        logger.success(f"   ✅ Patient name validation passed")
                        
                        # Step 3: Process dates for patient
                        logger.progress("\n🔄 Step 3: Processing patient dates...")
                        patient_data = process_dates_for_patient(patient_data, doc_id)
                        
                        if not patient_data:
                            logger.warning(f"   ❌ Skipping - insufficient date information")
                            api_results['error_message'] = "Insufficient date information"
                            error_count += 1
                            # Save API details for date processing failure
                            save_api_push_details(doc_id, {}, {}, None, api_results, api_writer)
                            continue
                        
                        logger.success(f"   ✅ Date processing completed")
                        logger.data("Processed Patient Data", patient_data)
                        
                        # Step 4: Extract order data
                        logger.progress("\n🔄 Step 4: Extracting order data...")
                        order_data = extract_order_data(text)
                        
                        # Clean order data
                        order_data = remove_none_fields(order_data)
                        
                        # Step 4.5: Merge patient and order data for completeness
                        logger.progress("\n🔄 Step 4.5: Merging patient and order data...")
                        patient_data = merge_patient_order_data(patient_data, order_data)
                        logger.success(f"   ✅ Data merged successfully")
                        
                        order_data["companyId"] = company_map.get(agency.lower())
                        order_data["pgCompanyId"] = PG_ID
                        order_data["documentID"] = doc_id
                        order_data["sentToPhysicianDate"] = received
                        order_data["signedByPhysicianDate"] = fetch_signed_date(doc_id)
                        
                        # Validate company ID
                        if order_data["companyId"] == "" or order_data["companyId"] == None:
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Missing companyID"])
                            logger.warning(f"   ❌ Skipping - missing company ID for agency: {agency}")
                            api_results['error_message'] = f"Missing company ID for agency: {agency}"
                            error_count += 1
                            # Save API details for missing company ID
                            save_api_push_details(doc_id, patient_data, order_data, None, api_results, api_writer)
                            continue
                        
                        # Process order dates
                        order_data = process_dates_for_order(order_data, "")
                        
                        # Set order date if missing
                        if not order_data.get("orderDate"):
                            order_data["orderDate"] = received
                        
                        # Validate required fields
                        if not order_data.get("episodeStartDate"):
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Missing episodeStartDate"])
                            logger.warning(f"   ❌ Skipping - missing episode start date")
                            api_results['error_message'] = "Missing episode start date"
                            error_count += 1
                            # Save API details for missing episode start date
                            save_api_push_details(doc_id, patient_data, order_data, None, api_results, api_writer)
                            continue
                        elif not order_data.get("episodeEndDate"):
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Missing episodeEndDate"])
                            logger.warning(f"   ❌ Skipping - missing episode end date")
                            api_results['error_message'] = "Missing episode end date"
                            error_count += 1
                            # Save API details for missing episode end date
                            save_api_push_details(doc_id, patient_data, order_data, None, api_results, api_writer)
                            continue
                        elif not order_data.get("startOfCare"):
                            with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([doc_id, "Missing startOfCare"])
                            logger.warning(f"   ❌ Skipping - missing start of care date")
                            api_results['error_message'] = "Missing start of care date"
                            error_count += 1
                            # Save API details for missing start of care
                            save_api_push_details(doc_id, patient_data, order_data, None, api_results, api_writer)
                            continue
                        
                        logger.success(f"   ✅ Order data extracted and validated")
                        
                        # Step 5: Check if patient exists or create new patient
                        logger.progress("\n🔄 Step 5: Checking/Creating patient...")
                        patient_id = check_patient_exists_comprehensive(patient_data, doc_id)
                        
                        if not patient_id:
                            # Create new patient
                            logger.info("Patient not found, creating new patient via API...", doc_id)
                            patient_id = create_patient_via_api(patient_data, doc_id)
                            if patient_id:
                                api_results['patient_created'] = True
                                logger.success(f"   ✅ New patient created with ID: {patient_id}")
                            else:
                                logger.error(f"   ❌ Failed to create patient")
                                api_results['error_message'] = "Failed to create patient via API"
                                api_error_count += 1
                                # Save API details for failed patient creation
                                save_api_push_details(doc_id, patient_data, order_data, None, api_results, api_writer)
                                continue
                        else:
                            logger.success(f"   ✅ Existing patient found with ID: {patient_id}")
                        
                        # Step 6: Push order to API
                        logger.progress("\n🔄 Step 6: Pushing order to API...")
                        order_push_result = push_order_via_api(order_data, patient_id, doc_id)
                        
                        if order_push_result["success"]:
                            api_results['order_pushed'] = True
                            api_results['status'] = 'SUCCESS'
                            logger.success(f"   ✅ Order pushed successfully")
                            api_success_count += 1
                        else:
                            logger.error(f"   ❌ Failed to push order")
                            api_results['error_message'] = "Failed to push order via API"
                            
                            # Check for duplicate order in the response
                            if order_push_result["status_code"] == 409:
                                try:
                                    import json
                                    response_data = json.loads(order_push_result["response"])
                                    if "orderId" in response_data:
                                        api_results['duplicate_order_id'] = response_data["orderId"]
                                except:
                                    # If JSON parsing fails, try to extract order ID from text
                                    response_text = order_push_result["response"]
                                    if "orderId" in response_text:
                                        order_id_match = re.search(r'"orderId":"([^"]+)"', response_text)
                                        if order_id_match:
                                            api_results['duplicate_order_id'] = order_id_match.group(1)
                            
                            api_error_count += 1
                        
                        # Step 7: Write to CSV
                        logger.progress("\n🔄 Step 7: Writing to CSV...")
                        if write_to_csv(patient_data, order_data, doc_id, agency, csv_writer):
                            success_count += 1
                            logger.success(f"   ✅ Document {doc_id} processed and saved successfully")
                        else:
                            error_count += 1
                            logger.error(f"   ❌ Failed to write data for document {doc_id}")
                        
                        # Step 8: Save API push details (ALWAYS executed for successful processing)
                        logger.progress("\n🔄 Step 8: Saving API details...")
                        save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, api_writer)
                        logger.success(f"   ✅ API details saved")
                            
                    except Exception as e:
                        logger.error(f"   ❌ Critical error: {str(e)}")
                        api_results['error_message'] = str(e)
                        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([doc_id, f"Critical error: {str(e)}"])
                        error_count += 1
                        api_error_count += 1
                        
                        # ALWAYS save API details even on critical error
                        save_api_push_details(doc_id, patient_data if 'patient_data' in locals() else {}, 
                                            order_data if 'order_data' in locals() else {}, 
                                            patient_id, api_results, api_writer)
                        continue
    
    except Exception as e:
        logger.error(f"Critical error in CSV processing: {str(e)}")
    
    # Final summary
    logger.header("PROCESSING SUMMARY")
    logger.info(f"Total Processed: {processed_count}")
    logger.info(f"CSV Success: {success_count}")
    logger.info(f"CSV Errors: {error_count}")
    logger.info(f"API Success: {api_success_count}")
    logger.info(f"API Errors: {api_error_count}")
    logger.info(f"CSV Success Rate: {(success_count/processed_count*100):.1f}%" if processed_count > 0 else "0%")
    logger.info(f"API Success Rate: {(api_success_count/processed_count*100):.1f}%" if processed_count > 0 else "0%")
    logger.info(f"Output CSV File: {output_filename}")
    logger.info(f"API Details File: {api_details_filename}")
    logger.header("="*60)

def normalize_date_string(date_str):
    """Normalize date string to MM/DD/YYYY format"""
    try:
        date_str = safe_strip(date_str).replace("-", "/")
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%m/%d/%Y")
    except Exception:
        return ""

def merge_patient_order_data(patient_data, order_data):
    """Merge patient and order data to get the most complete information"""
    try:
        # Prioritize the best MRN found (longer, more complete ones preferred)
        patient_mrn = patient_data.get("medicalRecordNo", "")
        order_mrn = order_data.get("mrn", "")
        
        # Choose the better MRN (longer and more complete)
        if not patient_mrn and order_mrn:
            patient_data["medicalRecordNo"] = order_mrn
            logger.info(f"Using MRN from order data: '{order_mrn}'")
        elif not order_mrn and patient_mrn:
            logger.info(f"Using MRN from patient data: '{patient_mrn}'")
        elif patient_mrn and order_mrn:
            # If both exist, prefer the longer/more complete one
            if len(order_mrn) > len(patient_mrn):
                patient_data["medicalRecordNo"] = order_mrn
                logger.info(f"Using longer MRN from order data: '{order_mrn}' over '{patient_mrn}'")
            else:
                logger.info(f"Using MRN from patient data: '{patient_mrn}' over '{order_mrn}'")
        else:
            logger.warning("No MRN found in either patient or order data")
        
        # Handle sex field - use order data as fallback
        patient_sex = patient_data.get("patientSex", "")
        order_sex = order_data.get("patientSex", "")
        
        if not patient_sex and order_sex:
            patient_data["patientSex"] = order_sex
        elif patient_sex and order_sex and patient_sex != order_sex:
            logger.warning(f"Sex mismatch: Patient='{patient_sex}' vs Order='{order_sex}', using patient data")
        
        # Final fallback: try to infer sex from first name if still empty
        if not patient_data.get("patientSex"):
            inferred_sex = infer_sex_from_name(patient_data.get("patientFName", ""))
            if inferred_sex:
                patient_data["patientSex"] = normalize_sex(inferred_sex)
            # Only show warning if ALL methods failed and we have a patient name
            elif patient_data.get("patientFName"):
                logger.warning("Could not determine sex using any available method")
        
        # Handle service line field - merge from both sources
        patient_service = patient_data.get("serviceLine", "")
        order_service = order_data.get("serviceLine", "")
        
        if not patient_service and order_service:
            patient_data["serviceLine"] = order_service
            logger.info(f"Using service line from order data: '{order_service}'")
        elif patient_service and not order_service:
            logger.info(f"Using service line from patient data: '{patient_service}'")
        elif patient_service and order_service:
            # Both exist, combine them if different
            if patient_service.lower() != order_service.lower():
                # Combine unique services
                patient_services = set([s.strip() for s in patient_service.split(',') if s.strip()])
                order_services = set([s.strip() for s in order_service.split(',') if s.strip()])
                combined_services = patient_services.union(order_services)
                patient_data["serviceLine"] = ', '.join(sorted(combined_services))
                logger.info(f"Combined service lines: Patient='{patient_service}' + Order='{order_service}' = '{patient_data['serviceLine']}'")
            else:
                logger.info(f"Service line consistent in both sources: '{patient_service}'")
        
        # Handle payer source field - use order data as fallback
        patient_payer = patient_data.get("payorSource", "")
        order_payer = order_data.get("payorSource", "")
        
        if not patient_payer and order_payer:
            patient_data["payorSource"] = order_payer
            logger.info(f"Using payer source from order data: '{order_payer}'")
        elif patient_payer and not order_payer:
            logger.info(f"Using payer source from patient data: '{patient_payer}'")
        elif patient_payer and order_payer:
            # Both exist, prefer patient data but log if they differ
            if patient_payer.lower() != order_payer.lower():
                logger.warning(f"Payer source mismatch: Patient='{patient_payer}' vs Order='{order_payer}', using patient data")
            else:
                logger.info(f"Payer source consistent in both sources: '{patient_payer}'")
        
        # Use order data to fill missing patient fields
        if not patient_data.get("address") and order_data.get("patientAddress"):
            patient_data["address"] = order_data["patientAddress"]
            
        if not patient_data.get("city") and order_data.get("patientCity"):
            patient_data["city"] = order_data["patientCity"]
            
        if not patient_data.get("state") and order_data.get("patientState"):
            patient_data["state"] = order_data["patientState"]
            
        if not patient_data.get("zip") and order_data.get("patientZip"):
            patient_data["zip"] = normalize_zip(order_data["patientZip"])
            
        if not patient_data.get("phoneNumber") and order_data.get("patientPhone"):
            patient_data["phoneNumber"] = order_data["patientPhone"]
        
        # Cross-validate patient names
        if order_data.get("patientName") and patient_data.get("patientFName"):
            order_name = safe_strip(order_data["patientName"]).lower()
            patient_fname = safe_strip(patient_data["patientFName"]).lower()
            patient_lname = safe_strip(patient_data["patientLName"]).lower()
            
            if patient_fname in order_name and patient_lname in order_name:
                logger.success("Patient name cross-validation successful")
            else:
                logger.warning(f"Patient name mismatch: Order='{order_data['patientName']}' vs Patient='{patient_data['patientFName']} {patient_data['patientLName']}'")
        
        return patient_data
        
    except Exception as e:
        logger.warning(f"Error merging patient and order data: {str(e)}")
        return patient_data

def process_dates_for_order(order_data, patient_id):
    """Process and validate dates for order data"""
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

    # Fill missing dates
    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    if not soe_dt and eoe_dt:
        soe_dt = eoe_dt - timedelta(days=59)

    if not eoe_dt and soe_dt:
        eoe_dt = soe_dt + timedelta(days=59)
        
    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    # Update order data
    order_data["startOfCare"] = format_date(soc_dt)
    order_data["episodeStartDate"] = format_date(soe_dt)
    order_data["episodeEndDate"] = format_date(eoe_dt)

    return order_data

def main():
    """Main execution function"""
    
    logger.header("FINAL VERSION PROCESSOR")
    logger.info("CSV Data Extraction & API Push Tool")
    logger.info("")
    logger.info("Features:")
    logger.info("- 2 Robust text extraction methods (PDFPlumber + Tesseract OCR)")
    logger.info("- AI-powered patient and order data extraction")
    logger.info("- Enhanced error handling and logging")
    logger.info("- Full API integration (patient creation + order pushing)")
    logger.info("- CSV output with comprehensive patient data")
    logger.info("- Detailed API tracking with remarks for every document")
    logger.info("- Complete audit trail of all processing attempts")
    logger.info("")
    
    try:
        # Process the CSV file
        process_csv("hawthorn_fam.csv")
    finally:
        # Ensure logger is properly closed
        logger.close()

if __name__ == "__main__":
    main() 