import os
import base64
import pandas as pd
import google.generativeai as genai
import json
import uuid
from datetime import datetime
import time
import requests
import re
import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io

# Constants
PATIENTS_JSON_FILE = "patients.json"
PG_ID = "your_pg_id_here"  # Replace with actual PG ID
HEADERS = {
    # Add your headers here
}

def safe_strip(value, default=""):
    """Safely strip whitespace from a value"""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

def extract_text_pdfplumber(pdf_bytes):
    """Extract text using pdfplumber (primary method)"""
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = []
            # Limit to first 10 pages for performance
            for page_num in range(min(len(pdf.pages), 10)):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n".join(text_parts)
    except Exception as e:
        raise Exception(f"PDFPlumber extraction failed: {str(e)}")

def extract_text_tesseract_ocr(pdf_bytes):
    """Extract text using Tesseract OCR (fallback method)"""
    try:
        import io
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        
        # Limit to first 10 pages for performance
        for page_num in range(min(doc.page_count, 10)):
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

def get_pdf_text_from_file(pdf_path):
    """
    Read a local PDF file and return its text using the two-stage
    strategy (pdfplumber → Tesseract OCR fallback).
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Method 1: PDFPlumber (primary)
        try:
            text = extract_text_pdfplumber(pdf_bytes)
            if text and text.strip():
                Logger.success(f"Text extracted with pdfplumber ({len(text)} characters)")
                return text
            raise Exception("pdfplumber returned empty text")
        except Exception as e:
            Logger.warning(f"pdfplumber failed: {e}; trying OCR...")

        # Method 2: Tesseract OCR (fallback)
        try:
            text = extract_text_tesseract_ocr(pdf_bytes)
            if text and text.strip():
                Logger.success(f"Text extracted with OCR ({len(text)} characters)")
                return text
            raise Exception("OCR returned empty text")
        except Exception as e:
            Logger.error(f"All extraction methods failed: {e}")
            return None
            
    except Exception as e:
        Logger.error(f"Error reading PDF file {pdf_path}: {str(e)}")
        return None

def normalize_date_format(date_str):
    """Normalize date string to MM/DD/YYYY format, converting hyphens to slashes"""
    if not date_str or not isinstance(date_str, str):
        return ""
    
    try:
        # Replace hyphens with slashes
        normalized_date = date_str.strip().replace("-", "/")
        
        # Parse and reformat to ensure consistent MM/DD/YYYY format
        dt = datetime.strptime(normalized_date, "%m/%d/%Y")
        return dt.strftime("%m/%d/%Y")
    except ValueError:
        try:
            # Try other common formats and convert to MM/DD/YYYY
            for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                try:
                    dt = datetime.strptime(date_str.strip(), fmt)
                    return dt.strftime("%m/%d/%Y")
                except ValueError:
                    continue
        except:
            pass
    
    # If all parsing fails, return the original string with hyphens replaced by slashes
    return date_str.strip().replace("-", "/") if date_str else ""

class Logger:
    @staticmethod
    def success(message):
        print(f"✅ {message}")
        
    @staticmethod
    def warning(message):
        print(f"⚠️ {message}")
        
    @staticmethod
    def error(message):
        print(f"❌ {message}")

def save_patient_data(fname, lname, dob, patient_id):
    """Save patient data to local JSON file"""
    patient_key = f"{fname.lower()}_{lname.lower()}_{dob}"
    data = {
        patient_key: {
            "patient_id": patient_id,
            "first_name": fname,
            "last_name": lname,
            "dob": dob,
            "last_updated": datetime.now().isoformat()
        }
    }
    
    try:
        if os.path.exists(PATIENTS_JSON_FILE):
            with open(PATIENTS_JSON_FILE, 'r') as f:
                existing_data = json.load(f)
            existing_data.update(data)
            data = existing_data
            
        with open(PATIENTS_JSON_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        Logger.error(f"Error saving patient data: {str(e)}")

def check_if_patient_exists(fname, lname, dob):
    """Check if patient already exists in the system."""
    check_ids = [PG_ID]
    
    # Normalize the DOB to ensure consistent format
    dob = normalize_date_format(dob)
    fname = fname.strip().upper()
    lname = lname.strip().upper()
    
    # First check local JSON file
    patient_key = f"{fname.lower()}_{lname.lower()}_{dob}"
    if os.path.exists(PATIENTS_JSON_FILE):
        try:
            with open(PATIENTS_JSON_FILE, 'r') as f:
                patient_data = json.load(f)
                if patient_key in patient_data:
                    patient_id = patient_data[patient_key]["patient_id"]
                    Logger.success(f"Patient found in local cache: {fname} {lname}, DOB: {dob}, ID: {patient_id}")
                    return patient_id
        except Exception as e:
            Logger.warning(f"Error reading local patient data: {str(e)}")
    
    # Check in remote system
    for check_id in check_ids:
        url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/company/pg/{check_id}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                Logger.warning(f"Failed to fetch patient list for company ID: {check_id}")
                continue

            patients = response.json()
            for p in patients:
                info = p.get("agencyInfo", {})
                if not info:
                    continue
                existing_fname = (info.get("patientFName") or "").strip().upper()
                existing_lname = (info.get("patientLName") or "").strip().upper()
                existing_dob = normalize_date_format(info.get("dob") or "")
                
                if existing_fname == fname and existing_lname == lname and existing_dob == dob:
                    patient_id = p.get("id") or p.get("patientId")
                    Logger.success(f"Patient exists in company {check_id}: {fname} {lname}, DOB: {dob}, ID: {patient_id}")
                    # Save to local JSON file
                    save_patient_data(fname, lname, dob, patient_id)
                    return patient_id
                    
        except requests.exceptions.RequestException as e:
            Logger.error(f"Network error checking patient existence: {str(e)}")
        except Exception as e:
            Logger.error(f"Error checking patient existence: {str(e)}")
            
    return None

def extract_info_with_gemini(model, pdf_path, filename):
    """Extract information from PDF using Gemini with robust text extraction"""
    try:
        # Extract text using the robust two-stage method
        text = get_pdf_text_from_file(pdf_path)
        if not text:
            print(f"Could not extract text from {filename}")
            return None
        
        prompt = """
        You are a medical document analysis expert. Your primary task is to extract specific information from medical documents.
        Focus especially on order numbers, MRN, and dates. Extract ONLY information that is EXPLICITLY present.

        CRITICAL FIELDS TO EXTRACT (with exact phrases to look for):

        1. Order Number [CRITICAL]:
           - Look for: "Order #:", "Order Number:", "Reference #:", "Ref #:", "Document #:"
           - This is usually at the top of the document
           - It's often a numeric value, may include letters
           - Example formats: "Order #: 12345", "Ref#: ABC-12345"

        2. MRN (Medical Record Number) [CRITICAL]:
           - Look for: "MRN:", "Medical Record #:", "Medical Record Number:", "Chart #:"
           - Usually near patient information
           - Can be numeric or alphanumeric
           - Example formats: "MRN: 123456", "Medical Record #: ABC123"

        3. Dates [CRITICAL] - Look for these exact phrases:
           a) Order Date:
              - "Order Date:", "Date:", "Created Date:", "Document Date:"
           
           b) Start of Care:
              - "Start of Care:", "SOC Date:", "Initial Start Date:"
              - "Care Start Date:", "Begin Care Date:"
           
           c) Episode Start Date:
              - "Certification Period From:", "Episode Start Date:"
              - "Cert Period From:", "Period Start Date:"
              - "Certification Start Date:"
           
           d) Episode End Date:
              - "Certification Period To:", "Episode End Date:"
              - "Cert Period To:", "Period End Date:"
              - "Certification End Date:"

        4. Patient Information:
           - First Name and Last Name (separately)
           - DOB (Date of Birth): "DOB:", "Birth Date:", "Date of Birth:"

        5. Document Status:
           - Sent to Physician Date: Look for "Sent to MD:", "Sent for Signature:", "Sent Date:"
           - Signed by Physician Date: Look for "Signed Date:", "MD Signature Date:", "Provider Signed:"
           - Document Name/Type: Look for "Form Type:", "Document Type:", "Form Name:"

        CRITICAL INSTRUCTIONS:
        1. For Order Numbers and MRN:
           - Extract EXACT values as shown
           - Include any prefix/suffix numbers or letters
           - Do not modify or format the numbers

        2. For Dates:
           - Convert all dates to MM/DD/YYYY format using FORWARD SLASHES only
           - NEVER use hyphens (-) in dates, always use forward slashes (/)
           - Pad single digits with zeros (e.g., "4/6/2025" → "04/06/2025")
           - Only extract dates that are clearly labeled
           - If a date is not explicitly present, return empty string
           - Do not infer dates from context

        3. For Status Fields:
           - Set to true only if explicitly confirmed
           - Look for signatures, stamps, or digital signature indicators
           - Set to false if pending or no clear indication

        Return this exact JSON format:
        {
            "orderNo": "",           // CRITICAL - Extract exactly as shown
            "orderDate": "",         // MM/DD/YYYY format with forward slashes
            "mrn": "",              // CRITICAL - Extract exactly as shown
            "startOfCare": "",      // MM/DD/YYYY format with forward slashes
            "episodeStartDate": "", // MM/DD/YYYY format with forward slashes
            "episodeEndDate": "",   // MM/DD/YYYY format with forward slashes
            "patientFirstName": "", 
            "patientLastName": "",
            "patientDOB": "",       // MM/DD/YYYY format with forward slashes
            "sentToPhysicianDate": "",
            "sentToPhysicianStatus": false,
            "signedByPhysicianDate": "",
            "signedByPhysicianStatus": false,
            "uploadedSignedOrderStatus": false,
            "documentName": ""
        }

        IMPORTANT: Double-check extracted order numbers, MRN, and dates before returning.
        If these critical fields are not found with their exact labels, return empty string.
        Do not attempt to infer or generate any missing information.
        """
        
        # Send extracted text to Gemini instead of PDF bytes
        response = model.generate_content(
            [prompt, text],
            stream=False
        )
        
        response_text = response.text.strip()
        print(f"\nProcessed {filename}, response length:", len(response_text))
        
        # Extract JSON from response
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            print(f"No JSON found in response for {filename}")
            print("Raw response:", response_text)
            return None
            
        data = json.loads(match.group())
        
        # Normalize all date fields to ensure consistent MM/DD/YYYY format
        date_fields = ['orderDate', 'startOfCare', 'episodeStartDate', 'episodeEndDate', 
                      'patientDOB', 'sentToPhysicianDate', 'signedByPhysicianDate']
        
        for field in date_fields:
            if field in data:
                data[field] = normalize_date_format(data[field])
        
        # Validate critical fields
        critical_fields = ['orderNo', 'mrn', 'startOfCare', 'episodeStartDate', 'episodeEndDate']
        missing_fields = [field for field in critical_fields if not data.get(field)]
        if missing_fields:
            print("\nWarning: Missing critical fields:", ', '.join(missing_fields))
        
        # Add order ID (GUID)
        data['orderId'] = str(uuid.uuid4())
        
        # Look up patient ID if we have name and DOB
        if data['patientFirstName'] and data['patientLastName'] and data['patientDOB']:
            patient_id = check_if_patient_exists(
                data['patientFirstName'],
                data['patientLastName'],
                data['patientDOB']
            )
            data['patientId'] = patient_id if patient_id else ''
            
            if not patient_id:
                print(f"No existing patient found for: {data['patientFirstName']} {data['patientLastName']}, DOB: {data['patientDOB']}")
        else:
            data['patientId'] = ''
            print("Missing patient name or DOB - cannot look up patient ID")
        
        # Print extracted information
        print("\nExtracted information:")
        print(json.dumps(data, indent=2))
        
        return data
        
    except Exception as e:
        print(f"Error extracting information from {filename}: {str(e)}")
        return None

def process_pdfs(folder_path, api_key, company_json_path, entities_csv_path):
    """Process all PDFs in the folder and create Excel output"""
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Load company and entities data
    with open(company_json_path, 'r') as f:
        company_data = json.load(f)
    
    # Read entities from CSV instead of Excel
    try:
        entities_data = pd.read_csv(entities_csv_path)
        print(f"Loaded {len(entities_data)} entities from CSV")
    except Exception as e:
        print(f"Warning: Could not load entities CSV: {str(e)}")
        entities_data = None

    results = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\nProcessing file {i} of {len(pdf_files)}: {pdf_file}")
        
        try:
            # Extract information using Gemini
            info = extract_info_with_gemini(model, pdf_path, pdf_file)
            if not info:
                print(f"Could not extract information from {pdf_file}")
                continue
            
            # Add company and entity information
            info['companyId'] = company_data.get('companyId', '')
            
            # Look up PG Company ID from entities CSV
            if entities_data is not None:
                # Add logic to match and get pgCompanyId
                info['pgCompanyId'] = ''  # Update this based on your matching logic
            
            results.append(info)
            print(f"Successfully processed {pdf_file}")
            
            # Save progress after each file
            temp_df = pd.DataFrame(results)
            temp_df.to_excel('output_results_temp.xlsx', index=False)
            print("Progress saved to temporary file")
            
            # Add delay between files to avoid rate limits
            if i < len(pdf_files):
                delay = 10
                print(f"Waiting {delay} seconds before next file...")
                time.sleep(delay)
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    if results:
        # Save final results
        df = pd.DataFrame(results)
        df.to_excel('adobe_processing_results.xlsx', index=False)
        print("\nFinal results saved to adobe_processing_results.xlsx")
        return df
    else:
        print("No results to save")
        return None

if __name__ == "__main__":
    FOLDER_PATH = r"C:\Users\pujit\Vivnovation\faxed\adobe\may\may"
    API_KEY = "AIzaSyA9FX4T6UeMldAA_KoynJ75OFc7jR57vtU"
    COMPANY_JSON_PATH = r"C:\Users\pujit\Vivnovation\faxed\adobe\company.json"
    ENTITIES_CSV_PATH = r"C:\Users\pujit\Vivnovation\faxed\adobe\entities.csv"
    
    print("Processing PDFs...")
    df = process_pdfs(FOLDER_PATH, API_KEY, COMPANY_JSON_PATH, ENTITIES_CSV_PATH)
    print("\nProcessing complete")
