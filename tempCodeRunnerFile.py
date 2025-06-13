import csv
import json
import base64
import io
import re
import os
from datetime import datetime, timedelta
import requests
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import fitz
from PIL import Image
import pytesseract

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("api_outputs", exist_ok=True)
os.makedirs("csv_outputs", exist_ok=True)

# Enhanced Logger from final_version.py
class Logger:
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, f"processing_log_{timestamp}.txt"), "w", encoding='utf-8')
    def _write_to_file(self, message):
        if self.log_file:
            clean_message = re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', message)
            self.log_file.write(clean_message + "\n")
            self.log_file.flush()
    def _colorize(self, text, color):
        return f"{Logger.COLORS.get(color, '')}{text}{Logger.COLORS['RESET']}"
    def _get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S")
    def header(self, title):
        message = f"\n{'='*80}\n{title.center(80)}\n{'='*80}"
        print(self._colorize(message, 'CYAN'))
        self._write_to_file(message)
    def info(self, message, doc_id=None):
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'BLUE')} ℹ️  {message}"
        print(message)
        self._write_to_file(message)
    def success(self, message, doc_id=None):
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'GREEN')} ✅ {self._colorize(message, 'GREEN')}"
        print(message)
        self._write_to_file(message)
    def error(self, message, doc_id=None):
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'RED')} ❌ {self._colorize(message, 'RED')}"
        print(message)
        self._write_to_file(message)
    def warning(self, message, doc_id=None):
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'YELLOW')} ⚠️  {self._colorize(message, 'YELLOW')}"
        print(message)
        self._write_to_file(message)
    def progress(self, message, doc_id=None):
        timestamp = self._get_timestamp()
        prefix = f"[{timestamp}]"
        if doc_id:
            prefix += f" [DOC: {doc_id}]"
        message = f"{self._colorize(prefix, 'PURPLE')} 🔄 {message}"
        print(message)
        self._write_to_file(message)
    def data(self, title, data, doc_id=None):
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
    def close(self):
        if self.log_file:
            self.log_file.close()

logger = Logger()

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TOKEN = os.getenv("AUTH_TOKEN")
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
DOC_STATUS_URL = "https://api.doctoralliance.com/document/get?docId.id="
PATIENT_CREATE_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/create"
ORDER_PUSH_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Order"



PG_ID = "d10f46ad-225d-4ba2-882c-149521fcead5"  
PG_NAME = "Prima Care"
PG_NPI = "1265422596"

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

def is_scanned_pdf(pdf_bytes):
    """Check if a PDF is scanned by analyzing its content"""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Check first few pages
            for page in pdf.pages[:3]:
                text = page.extract_text()
                # If text is None or very short, likely scanned
                if not text or len(text.strip()) < 50:
                    return True
        return False
    except Exception as e:
        logger.warning(f"Error checking if PDF is scanned: {e}")
        return True  # Assume scanned if check fails

def extract_text_with_ocr(pdf_bytes, doc_id):
    """Extract text from scanned PDF using OCR"""
    try:
        logger.progress("Using OCR to extract text from scanned document", doc_id)
        text_parts = []
        
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Extract text using pytesseract
            page_text = pytesseract.image_to_string(img)
            text_parts.append(page_text)
        
        pdf_document.close()
        full_text = "\n".join(text_parts)
        
        # Clean up the text
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        logger.data("OCR extracted text (truncated)", cleaned_text[:500], doc_id)
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {e}", doc_id)
        raise

def get_pdf_text(doc_id):
    logger.progress("Step 1: Fetching and extracting PDF text", doc_id)
    try:
        response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS)
        if response.status_code != 200:
            logger.error(f"Failed to fetch document: {response.text}", doc_id)
            raise Exception("Failed to fetch document")
            
        daId = response.json()["value"]["patientId"]["id"]
        document_buffer = response.json()["value"]["documentBuffer"]
        pdf_bytes = base64.b64decode(document_buffer)
        
        # Check if document is scanned
        if is_scanned_pdf(pdf_bytes):
            logger.info("Document appears to be scanned, using OCR", doc_id)
            text = extract_text_with_ocr(pdf_bytes, doc_id)
        else:
            logger.info("Document appears to be digital, using PDFPlumber", doc_id)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        # Clean up the text
        edited = re.sub(r'\b\d[A-Z][A-Z0-9]\d[A-Z][A-Z0-9]\d[A-Z]{2}(?:\d{2})?\b', '', text)
        logger.data("Extracted PDF text (truncated)", edited[:500], doc_id)
        return [edited, daId]
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}", doc_id)
        raise

def extract_patient_data(text, doc_id=None):
    query = """
You are a medical data extraction expert. Extract the following fields from the provided medical document text. Return ONLY a valid JSON object, no extra text. If a field is missing, use an empty string or null. Do not infer values. Use the field names exactly as shown.

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

Clarifications:
- Dates: Use MM/DD/YYYY.
- Do not infer values.
- Only extract what is explicitly present.
- If a field is not found, leave it blank or null.
- Return only the JSON object.
"""
    try:
        logger.progress("Step 2: Extracting patient data using Gemini", doc_id)
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        logger.data("Extracted patient data", data, doc_id)
        return data
    except Exception as e:
        logger.error(f"Error extracting patient data: {e}", doc_id)
        return {}

def extract_order_data(text, doc_id=None):
    query = """
You are a medical data extraction expert. Extract the following fields from the provided medical document text. Return ONLY a valid JSON object, no extra text. If a field is missing, use an empty string or null. Do not infer values. Use the field names exactly as shown.

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
  "signedByPhysicianDate": "",
  "signedByPhysicianStatus": false,
  "patientId": "",
  "companyId": "",
  "pgCompanyId": "",
  "bit64Url": "",
  "documentName": "",
  "serviceLine": "",
  "payorSource": "",
  "patientSex": "",
  "patientAddress": "",
  "patientCity": "",
  "patientState": "",
  "patientZip": "",
  "patientPhone": ""
}

Clarifications:
- Dates: Use MM/DD/YYYY.
- Do not infer values.
- Only extract what is explicitly present.
- If a field is not found, leave it blank or null.
- Return only the JSON object.
"""
    try:
        logger.progress("Step 3: Extracting order data using Gemini", doc_id)
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        logger.data("Extracted order data", data, doc_id)
        return data
    except Exception as e:
        logger.error(f"Error extracting order data: {e}", doc_id)
        return {}

def fetch_signed_date(doc_id):
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
        logger.warning(f"Error fetching signed date: {e}", doc_id)
        return None

def get_patient_details_from_api(patient_id):
    try:
        url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/get-patient/{patient_id}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to fetch patient details for patient ID: {patient_id}")
            return {}
    except Exception as e:
        logger.warning(f"Error fetching patient details: {e}")
        return {}

def check_if_patient_exists(fname, lname, dob, doc_id=None):
    try:
        url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/company/pg/{PG_ID}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            logger.warning("Failed to fetch patient list.", doc_id)
            return None
        patients = response.json()
        dob = dob.strip()
        fname = fname.strip().upper()
        lname = lname.strip().upper()
        for p in patients:
            info = p.get("agencyInfo", {})
            if not info:
                continue
            existing_fname = (info.get("patientFName") or "").strip().upper()
            existing_lname = (info.get("patientLName") or "").strip().upper()
            existing_dob = (info.get("dob") or "").strip()
            if existing_fname == fname and existing_lname == lname and existing_dob == dob:
                patient_id = p.get("id") or p.get("patientId")
                logger.success(f"Patient exists: {fname} {lname}, DOB: {dob}, ID: {patient_id}", doc_id)
                return patient_id
        return None
    except Exception as e:
        logger.error(f"Error checking if patient exists: {e}", doc_id)
        return None

def get_or_create_patient(patient_data, daId, agency, doc_id=None):
    try:
        dob = patient_data.get("dob", "").strip()
        fname = patient_data.get("patientFName", "").strip().upper()
        lname = patient_data.get("patientLName", "").strip().upper()
        key = f"{fname}_{lname}_{dob}"
        if key in created_patients:
            logger.success(f"Patient already created earlier in this run: {key}", doc_id)
            return created_patients[key]
        existing_id = check_if_patient_exists(fname, lname, dob, doc_id)
        if existing_id:
            logger.success(f"Patient exists on platform: {key}, ID: {existing_id}", doc_id)
            created_patients[key] = existing_id
            with open("hawthorn_internalmedicine.json", "w") as f:
                json.dump(created_patients, f, indent=2)
            return existing_id
        # Add required fields
        patient_data["daBackofficeID"] = str(daId)
        patient_data["pgCompanyId"] = PG_ID
        patient_data["companyId"] = company_map.get(agency.strip().lower())
        patient_data["physicianGroup"] = PG_NAME
        patient_data["physicianGroupNPI"] = PG_NPI
        logger.info(f"Patient JSON for creating patient: {patient_data}", doc_id)
        resp = requests.post(PATIENT_CREATE_URL, headers={"Content-Type": "application/json"}, json=patient_data)
        logger.info(f"Patient creation status code: {resp.status_code}", doc_id)
        logger.info(f"Patient creation response: {resp.text}", doc_id)
        if resp.status_code == 201:
            new_id = resp.json().get("id") or resp.text
            created_patients[key] = new_id
            with open("hawthorn_internalmedicine.json", "w") as f:
                json.dump(created_patients, f, indent=2)
            return new_id
        else:
            logger.error(f"Failed to create patient: {resp.text}", doc_id)
        return None
    except Exception as e:
        logger.error(f"Error in get_or_create_patient: {e}", doc_id)
        return None

def push_order(order_data, doc_id, patient_id, agency, received, doc_signed_date):
    try:
        order_data["companyId"] = company_map.get(agency.lower())
        order_data["pgCompanyId"] = PG_ID
        order_data["patientId"] = patient_id
        order_data["documentID"] = doc_id
        order_data["sentToPhysicianDate"] = received
        order_data["signedByPhysicianDate"] = doc_signed_date
        if order_data.get("orderNo") is None:
            order_data["orderNo"] = doc_id + "1"
        resp = requests.post(ORDER_PUSH_URL, headers={"Content-Type": "application/json"}, json=order_data)
        logger.info(f"Order push status code: {resp.status_code}", doc_id)
        logger.info(f"Order push response: {resp.text}", doc_id)
        if resp.status_code == 201:
            logger.success(f"Order pushed successfully for patient ID: {patient_id}", doc_id)
            return True
        else:
            logger.error(f"Failed to push order: {resp.text}", doc_id)
            return False
    except Exception as e:
        logger.error(f"Error in push_order: {e}", doc_id)
        return False

def process_dates_for_patient(patient_data, doc_id):
    try:
        episode_info = patient_data.get("episodeDiagnoses", [{}])[0] or {}
        def safe_parse_date(date_str):
            try:
                date_str = str(date_str).strip()
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
        logger.info(f"Parsed Dates - SOC: {soc_dt}, SOE: {soe_dt}, EOE: {eoe_dt}", doc_id)
        if not soc_dt and soe_dt:
            soc_dt = soe_dt
        if not soe_dt and eoe_dt:
            soe_dt = eoe_dt - timedelta(days=59)
        if not eoe_dt and soe_dt:
            eoe_dt = soe_dt + timedelta(days=59)
        if not soe_dt and not eoe_dt and soc_dt:
            soe_dt = soc_dt
            eoe_dt = soe_dt + timedelta(days=59)
        if not soc_dt and not soe_dt and not eoe_dt:
            with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([doc_id, "Missing SOC, SOE, EOE"])
            logger.warning("Missing all episode dates", doc_id)
            return None
        episode_info["startOfCare"] = format_date(soc_dt)
        episode_info["startOfEpisode"] = format_date(soe_dt)
        episode_info["endOfEpisode"] = format_date(eoe_dt)
        patient_data["episodeDiagnoses"][0] = episode_info
        return patient_data
    except Exception as e:
        logger.error(f"Error in process_dates_for_patient: {e}", doc_id)
        return None

def process_dates_for_order(order_data, patient_id, doc_id):
    try:
        def parse_date(d):
            try:
                return datetime.strptime(d, "%m/%d/%Y") if d else None
            except ValueError:
                return None
        def format_date(d):
            return d.strftime("%m/%d/%Y") if d else ""
        soc = order_data.get("startOfCare")
        soe = order_data.get("episodeStartDate")
        eoe = order_data.get("episodeEndDate")
        soc_dt = parse_date(soc)
        soe_dt = parse_date(soe)
        eoe_dt = parse_date(eoe)
        patient_details = get_patient_details_from_api(patient_id)
        agency_info = patient_details.get("agencyInfo", {}) if patient_details else {}
        if not soc_dt and not soe_dt and not eoe_dt:
            soc_dt = parse_date(agency_info.get("startOfCare"))
            soe_dt = parse_date(agency_info.get("startOfEpisode"))
            eoe_dt = parse_date(agency_info.get("endOfEpisode"))
        if not soe_dt and not eoe_dt:
            soe_dt = parse_date(agency_info.get("startOfCare"))
            eoe_dt = soe_dt + timedelta(days=59) if soe_dt else None
        if not soc_dt and soe_dt:
            soc_dt = soe_dt
        if not soe_dt and eoe_dt:
            soe_dt = eoe_dt - timedelta(days=59)
        if not eoe_dt and soe_dt:
            eoe_dt = soe_dt + timedelta(days=59)
        if not soc_dt and eoe_dt:
            soc_dt = soe_dt
        order_data["startOfCare"] = format_date(soc_dt)
        order_data["episodeStartDate"] = format_date(soe_dt)
        order_data["episodeEndDate"] = format_date(eoe_dt)
        return order_data
    except Exception as e:
        logger.error(f"Error in process_dates_for_order: {e}", doc_id)
        return order_data

def write_to_csv(patient_data, order_data, doc_id, agency, csv_writer):
    """Write patient and order data to CSV file"""
    try:
        # Merge patient and order data for CSV output
        merged_data = merge_patient_order_data(patient_data, order_data)
        
        # Add document metadata
        merged_data.update({
            'ID': doc_id,
            'Name of Agency': agency,
            'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Created By': 'Automated Script',
            'Remarks': f'Processed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        })
        
        # Write to CSV
        csv_writer.writerow(merged_data)
        logger.success(f"Data written to CSV for Doc ID: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to CSV for Doc ID {doc_id}: {e}")
        return False

def save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, csv_writer):
    """Save API push details to tracking CSV"""
    try:
        api_row = {
            'Document_ID': doc_id,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Patient_ID': patient_id or '',
            'Patient_Created': api_results.get('patient_created', False),
            'Order_Pushed': api_results.get('order_pushed', False),
            'Patient_First_Name': patient_data.get('patientFName', ''),
            'Patient_Last_Name': patient_data.get('patientLName', ''),
            'Patient_DOB': patient_data.get('dob', ''),
            'Patient_Sex': patient_data.get('patientSex', ''),
            'Medical_Record_No': patient_data.get('medicalRecordNo', ''),
            'Service_Line': patient_data.get('serviceLine', ''),
            'Payer_Source': patient_data.get('payorSource', ''),
            'Physician_NPI': patient_data.get('physicianNPI', ''),
            'Agency_Name': patient_data.get('nameOfAgency', ''),
            'Patient_Address': patient_data.get('address', ''),
            'Patient_City': patient_data.get('city', ''),
            'Patient_State': patient_data.get('state', ''),
            'Patient_Zip': patient_data.get('zip', ''),
            'Patient_Phone': patient_data.get('phoneNumber', ''),
            'Patient_Email': patient_data.get('email', ''),
            'Order_Number': order_data.get('orderNo', ''),
            'Order_Date': order_data.get('orderDate', ''),
            'Start_Of_Care': order_data.get('startOfCare', ''),
            'Episode_Start_Date': order_data.get('episodeStartDate', ''),
            'Episode_End_Date': order_data.get('episodeEndDate', ''),
            'Sent_To_Physician_Date': order_data.get('sentToPhysicianDate', ''),
            'Signed_By_Physician_Date': order_data.get('signedByPhysicianDate', ''),
            'Company_ID': order_data.get('companyId', ''),
            'PG_Company_ID': order_data.get('pgCompanyId', ''),
            'SOC_Episode': patient_data.get('episodeDiagnoses', [{}])[0].get('startOfCare', ''),
            'Start_Episode': patient_data.get('episodeDiagnoses', [{}])[0].get('startOfEpisode', ''),
            'End_Episode': patient_data.get('episodeDiagnoses', [{}])[0].get('endOfEpisode', ''),
            'Diagnosis_1': patient_data.get('episodeDiagnoses', [{}])[0].get('firstDiagnosis', ''),
            'Diagnosis_2': patient_data.get('episodeDiagnoses', [{}])[0].get('secondDiagnosis', ''),
            'Diagnosis_3': patient_data.get('episodeDiagnoses', [{}])[0].get('thirdDiagnosis', ''),
            'Diagnosis_4': patient_data.get('episodeDiagnoses', [{}])[0].get('fourthDiagnosis', ''),
            'Diagnosis_5': patient_data.get('episodeDiagnoses', [{}])[0].get('fifthDiagnosis', ''),
            'Diagnosis_6': patient_data.get('episodeDiagnoses', [{}])[0].get('sixthDiagnosis', ''),
            'API_Status': api_results.get('status', 'FAILED'),
            'Error_Message': api_results.get('error_message', ''),
            'Remarks': f"Patient Created: {api_results.get('patient_created', False)}, Order Pushed: {api_results.get('order_pushed', False)}"
        }
        
        csv_writer.writerow(api_row)
        logger.success(f"API details saved for Doc ID: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving API details for Doc ID {doc_id}: {e}")
        return False

def merge_patient_order_data(patient_data, order_data):
    """Merge patient and order data for CSV output"""
    merged = {}
    
    # Patient data mapping
    if patient_data:
        merged.update({
            'Patient WAV ID': patient_data.get('patientWavId', ''),
            'Patient EHR Record ID': patient_data.get('patientEhrRecordId', ''),
            'Patient EHR Type': patient_data.get('patientEhrType', ''),
            'First Name': patient_data.get('patientFName', ''),
            'Middle Name': patient_data.get('patientMName', ''),
            'Last Name': patient_data.get('patientLName', ''),
            'Profile Picture': patient_data.get('profilePicture', ''),
            'Date of Birth': patient_data.get('dob', ''),
            'Age': patient_data.get('age', ''),
            'Sex': patient_data.get('patientSex', ''),
            'Status': patient_data.get('status', ''),
            'Marital Status': patient_data.get('maritalStatus', ''),
            'SSN': patient_data.get('ssn', ''),
            'Start of Care': patient_data.get('startOfCare', ''),
            'Medical Record No': patient_data.get('medicalRecordNo', ''),
            'Service Line': patient_data.get('serviceLine', ''),
            'Address': patient_data.get('address', ''),
            'City': patient_data.get('city', ''),
            'State': patient_data.get('state', ''),
            'Zip': patient_data.get('zip', ''),
            'Email': patient_data.get('email', ''),
            'Phone Number': patient_data.get('phoneNumber', ''),
            'Fax': patient_data.get('fax', ''),
            'Payor Source': patient_data.get('payorSource', ''),
            'Billing Provider': patient_data.get('billingProvider', ''),
            'Billing Provider Phone': patient_data.get('billingProviderPhone', ''),
            'Billing Provider Address': patient_data.get('billingProviderAddress', ''),
            'Billing Provider Zip': patient_data.get('billingProviderZip', ''),
            'NPI': patient_data.get('npi', ''),
            'Physician NPI': patient_data.get('physicianNPI', ''),
            'Supervising Provider': patient_data.get('supervisingProvider', ''),
            'Supervising Provider NPI': patient_data.get('supervisingProviderNPI', ''),
            'Physician Group': patient_data.get('physicianGroup', ''),
            'Physician Group NPI': patient_data.get('physicianGroupNPI', ''),
            'Physician Group Address': patient_data.get('physicianGroupAddress', ''),
            'Physician Phone': patient_data.get('physicianPhone', ''),
            'Physician Address': patient_data.get('physicianAddress', ''),
            'City State Zip': patient_data.get('cityStateZip', ''),
            'Patient Account No': patient_data.get('patientAccountNo', ''),
            'Agency NPI': patient_data.get('agencyNPI', ''),
            'Name of Agency': patient_data.get('nameOfAgency', ''),
            'Insurance ID': patient_data.get('insuranceId', ''),
            'Primary Insurance': patient_data.get('primaryInsurance', ''),
            'Secondary Insurance': patient_data.get('secondaryInsurance', ''),
            'Secondary Insurance ID': patient_data.get('secondaryInsuranceId', ''),
            'Tertiary Insurance': patient_data.get('tertiaryInsurance', ''),
            'Tertiary Insurance ID': patient_data.get('tertiaryInsuranceId', ''),
            'Next of Kin': patient_data.get('nextOfKin', ''),
            'Patient Caretaker': patient_data.get('patientCaretaker', ''),
            'Caretaker Contact Number': patient_data.get('caretakerContactNumber', ''),
            'DA Backoffice ID': patient_data.get('daBackofficeId', ''),
            'Company ID': patient_data.get('companyId', ''),
            'PG Company ID': patient_data.get('pgCompanyId', ''),
        })
        
        # Episode data
        if patient_data.get('episodeDiagnoses') and len(patient_data['episodeDiagnoses']) > 0:
            episode = patient_data['episodeDiagnoses'][0]
            merged.update({
                'Latest_Episode_StartOfCare': episode.get('startOfCare', ''),
                'Latest_Episode_StartOfEpisode': episode.get('startOfEpisode', ''),
                'Latest_Episode_EndOfEpisode': episode.get('endOfEpisode', ''),
                'Latest_Episode_FirstDiagnosis': episode.get('firstDiagnosis', ''),
                'Latest_Episode_SecondDiagnosis': episode.get('secondDiagnosis', ''),
                'Latest_Episode_ThirdDiagnosis': episode.get('thirdDiagnosis', ''),
                'Latest_Episode_FourthDiagnosis': episode.get('fourthDiagnosis', ''),
                'Latest_Episode_FifthDiagnosis': episode.get('fifthDiagnosis', ''),
                'Latest_Episode_SixthDiagnosis': episode.get('sixthDiagnosis', ''),
            })
    
    # Order data mapping
    if order_data:
        merged.update({
            'Line1 DOS From': order_data.get('episodeStartDate', ''),
            'Line1 DOS To': order_data.get('episodeEndDate', ''),
            'Line1 POS': order_data.get('placeOfService', ''),
            'Signed Orders': '1' if order_data.get('signedByPhysicianDate') else '0',
            'Total Orders': '1',
        })
    
    return merged

def process_csv(csv_path, max_rows=4):
    """Main CSV processing function - extracts data, pushes to API, and saves to output CSV"""
    
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
             open(api_details_filename, 'w', newline='', encoding='utf-8') as api_file, \
             open(csv_path, newline='', encoding='utf-8') as input_file:
            
            csv_writer = csv.DictWriter(output_file, fieldnames=csv_headers)
            csv_writer.writeheader()
            
            api_writer = csv.DictWriter(api_file, fieldnames=api_headers)
            api_writer.writeheader()
            
            logger.success(f"Created output CSV file: {output_filename}")
            logger.success(f"Created API details CSV file: {api_details_filename}")
            
            reader = csv.DictReader(input_file)
            i = 0
            
            for row in reader:
                if i >= max_rows:
                    break
                i += 1
                
                doc_id = row["ID"]
                agency = row["Facility"].strip()
                received = row["Received On"].strip()
                logger.info(f"Processing Document ID: {doc_id} for Agency: {agency} date : {received}", doc_id)
                
                # Initialize API results tracking
                api_results = {
                    'patient_created': False,
                    'order_pushed': False,
                    'status': 'FAILED',
                    'error_message': ''
                }
                patient_id = None
                
                try:
                    res = get_pdf_text(doc_id)
                    text = res[0]
                    if not text.strip():
                        raise ValueError("Empty text extracted from PDF")
                except Exception as e:
                    logger.error(f"Failed to extract text for Doc ID {doc_id}: {str(e)}", doc_id)
                    api_results['error_message'] = f"Could not extract text from PDF: {str(e)}"
                    with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, "Could not extract text from PDF"])
                    # Save API details for failed extraction
                    save_api_push_details(doc_id, {}, {}, None, api_results, api_writer)
                    continue
                
                patient_data = extract_patient_data(text, doc_id)
                logger.info(f"Response from gemini for patient: {patient_data}", doc_id)
                patient_data = process_dates_for_patient(patient_data, doc_id)
                logger.info(f"Patient data after setting dates : {patient_data}", doc_id)
                
                if not patient_data:
                    logger.error(f"Skipping patient creation for Doc ID {doc_id} due to insufficient date info.", doc_id)
                    api_results['error_message'] = "Insufficient patient date information"
                    save_api_push_details(doc_id, {}, {}, None, api_results, api_writer)
                    continue
                
                patient_id = get_or_create_patient(patient_data, res[1], agency, doc_id)
                logger.info(f"Patient Created : {patient_id}", doc_id)
                
                if patient_id:
                    api_results['patient_created'] = True
                    api_results['status'] = 'SUCCESS'
                
                order_data = extract_order_data(text, doc_id)
                order_data["companyId"] = company_map.get(agency.lower())
                order_data["pgCompanyId"] = PG_ID
                
                if not order_data["companyId"]:
                    with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, "Missing companyId"])
                    logger.error(f"Skipping order push for Doc ID {doc_id} due to missing companyID", doc_id)
                    api_results['error_message'] = "Agency not found in company mapping"
                    # Still write to CSV even if order push fails
                    write_to_csv(patient_data, order_data, doc_id, agency, csv_writer)
                    save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, api_writer)
                    continue
                
                order_data = process_dates_for_order(order_data, patient_id, doc_id)
                
                if not order_data.get("orderDate"):
                    order_data["orderDate"] = received
                elif not order_data.get("episodeStartDate"):
                    with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, "Missing episodeStartDate"])
                    logger.error(f"Skipping order push for Doc ID {doc_id} due to missing episodeStartDate", doc_id)
                    api_results['error_message'] = "Missing episodeStartDate"
                    write_to_csv(patient_data, order_data, doc_id, agency, csv_writer)
                    save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, api_writer)
                    continue
                elif not order_data.get("episodeEndDate"):
                    with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, "Missing episodeEndDate"])
                    logger.error(f"Skipping order push for Doc ID {doc_id} due to missing episodeEndDate", doc_id)
                    api_results['error_message'] = "Missing episodeEndDate"
                    write_to_csv(patient_data, order_data, doc_id, agency, csv_writer)
                    save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, api_writer)
                    continue
                elif not order_data.get("startOfCare"):
                    with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([doc_id, "Missing startOfCare"])
                    logger.error(f"Skipping order push for Doc ID {doc_id} due to missing startOfCare", doc_id)
                    api_results['error_message'] = "Missing startOfCare"
                    write_to_csv(patient_data, order_data, doc_id, agency, csv_writer)
                    save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, api_writer)
                    continue
                
                doc_signed_date = fetch_signed_date(doc_id)
                result = push_order(order_data, doc_id, patient_id, agency, received, doc_signed_date)
                
                if result:
                    logger.success(f"Order Push Response: Success", doc_id)
                    api_results['order_pushed'] = True
                    api_results['status'] = 'SUCCESS'
                else:
                    logger.error(f"Order Push Response: Failed", doc_id)
                    api_results['error_message'] = "Order push failed"
                
                # Write to CSV regardless of API success/failure
                write_to_csv(patient_data, order_data, doc_id, agency, csv_writer)
                save_api_push_details(doc_id, patient_data, order_data, patient_id, api_results, api_writer)
                
        logger.success(f"Processing completed. Output files:")
        logger.success(f"- Patient data CSV: {output_filename}")
        logger.success(f"- API tracking CSV: {api_details_filename}")
                
    except Exception as e:
        logger.error(f"Critical error in CSV processing: {e}")

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
        process_csv("prima.csv", max_rows=1)
    finally:
        # Ensure logger is properly closed
        logger.close()

if __name__ == "__main__":
    main() 