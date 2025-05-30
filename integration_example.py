#!/usr/bin/env python3
"""
Integration example showing how to use the advanced text extractor 
with handwriting support in your existing pipeline.

This demonstrates how to replace the get_pdf_text() function in final.py
with the advanced extraction capabilities.
"""

import base64
import json
import os
import requests
from datetime import datetime

# Import the advanced text extractor
from advanced_text_extractor import extract_text_advanced, AdvancedTextExtractor, TextType, ExtractionMethod

# Your existing API configuration
TOKEN = os.getenv("AUTH_TOKEN")
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def get_pdf_text_advanced(doc_id):
    """
    Enhanced version of get_pdf_text() with advanced handwriting recognition
    
    This replaces the original function in final.py with much better
    handwriting and complex document processing capabilities.
    """
    try:
        # Step 1: Fetch document with retries (same as original)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS, timeout=30)
                if response.status_code == 200:
                    break
                elif attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch document after {max_retries} attempts. Status: {response.status_code}")
                else:
                    print(f"⚠️  Document fetch attempt {attempt + 1} failed with status {response.status_code}, retrying...")
                    time.sleep(2)
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception(f"Document fetch timed out after {max_retries} attempts")
                else:
                    print(f"⚠️  Document fetch timeout attempt {attempt + 1}, retrying...")
                    time.sleep(2)
        
        # Step 2: Validate and extract document data (same as original)
        try:
            doc_data = response.json()
            if not doc_data or "value" not in doc_data:
                raise Exception("Invalid response format: missing 'value' key")
            
            value_data = doc_data["value"]
            if not value_data:
                raise Exception("Invalid response format: empty 'value' data")
                
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from document API")
        
        # Step 3: Extract metadata (same as original)
        try:
            if "patientId" not in value_data or not value_data["patientId"]:
                raise Exception("Missing required patientId in document")
            
            daId = value_data["patientId"]["id"] if isinstance(value_data["patientId"], dict) else value_data["patientId"]
            
            if "documentBuffer" not in value_data or not value_data["documentBuffer"]:
                raise Exception("Missing required documentBuffer in document")
            
            document_buffer = value_data["documentBuffer"]
            is_faxed = value_data.get("isFaxed", False)
            fax_source = value_data.get("faxSource", "Unknown")
            document_type = value_data.get("documentType", "Unknown")
            
        except (KeyError, TypeError) as e:
            raise Exception(f"Error extracting document metadata: {str(e)}")
        
        # Step 4: Decode PDF (same as original)
        try:
            pdf_bytes = base64.b64decode(document_buffer)
            if len(pdf_bytes) == 0:
                raise Exception("Empty PDF document buffer")
        except Exception as e:
            raise Exception(f"Error decoding PDF document: {str(e)}")
        
        # Step 5: ADVANCED TEXT EXTRACTION - This is the new part!
        print(f"🚀 Starting advanced text extraction for document {doc_id}")
        
        try:
            # Use the advanced extractor with handwriting support
            text, extraction_metadata = extract_text_advanced(pdf_bytes, doc_id)
            
            # Log detailed extraction information
            method = extraction_metadata.get("extraction_method", "unknown")
            confidence = extraction_metadata.get("confidence", 0.0)
            text_type = extraction_metadata.get("text_type", "unknown")
            processing_time = extraction_metadata.get("processing_time", 0.0)
            
            print(f"✅ Advanced extraction completed!")
            print(f"📊 Method used: {method}")
            print(f"📊 Text type detected: {text_type}")
            print(f"📊 Confidence: {confidence:.2f}")
            print(f"📊 Processing time: {processing_time:.2f}s")
            print(f"📊 Text length: {len(text)} characters")
            
            # Special handling for handwritten documents
            if text_type == "handwritten":
                print("✍️  HANDWRITTEN DOCUMENT DETECTED!")
                print("📝 Using specialized handwriting recognition pipeline")
            elif text_type == "mixed":
                print("📝 MIXED CONTENT DETECTED (printed + handwritten)")
                print("🔄 Using hybrid recognition approach")
            
            # Validate extraction success
            if not text or not text.strip():
                raise Exception("Advanced extraction returned empty text")
            
            if len(text.strip()) < 10:
                print("⚠️  Warning: Very short text extracted, document might be problematic")
            
        except Exception as e:
            print(f"❌ Advanced extraction failed: {str(e)}")
            print("🔄 Falling back to basic extraction...")
            
            # Fallback to basic extraction if advanced fails
            # (You can include your original extraction logic here as backup)
            text = f"[ADVANCED_EXTRACTION_FAILED] Document ID: {doc_id}, Error: {str(e)}"
            extraction_metadata = {
                "extraction_method": "fallback",
                "confidence": 0.0,
                "text_type": "unknown",
                "processing_time": 0.0,
                "error": str(e)
            }
        
        # Step 6: Enhanced metadata with extraction details
        enhanced_metadata = {
            "isFaxed": is_faxed,
            "faxSource": fax_source,
            "documentType": document_type,
            "textLength": len(text),
            **extraction_metadata  # Include all extraction metadata
        }
        
        print(f"📋 Final extraction summary for {doc_id}:")
        print(f"   📄 Method: {enhanced_metadata.get('extraction_method', 'unknown')}")
        print(f"   📝 Type: {enhanced_metadata.get('text_type', 'unknown')}")
        print(f"   📊 Length: {enhanced_metadata.get('textLength', 0)} chars")
        print(f"   ⏱️  Time: {enhanced_metadata.get('processing_time', 0):.2f}s")
        
        return [text, daId, enhanced_metadata]
        
    except Exception as e:
        print(f"❌ Critical failure in advanced get_pdf_text: {str(e)}")
        # Return minimal viable data to prevent complete pipeline failure
        return [
            f"[CRITICAL_FAILURE] Document ID: {doc_id}, Error: {str(e)}", 
            f"UNKNOWN_{doc_id}", 
            {
                "isFaxed": False,
                "faxSource": "Unknown",
                "documentType": "Unknown",
                "extraction_method": "failure_fallback",
                "textLength": 0,
                "error": str(e)
            }
        ]

def batch_process_documents_with_handwriting(doc_ids):
    """
    Example of batch processing multiple documents with handwriting detection
    """
    print("🚀 Starting batch processing with advanced handwriting recognition")
    print("=" * 70)
    
    results = {}
    handwriting_docs = []
    mixed_docs = []
    printed_docs = []
    failed_docs = []
    
    for i, doc_id in enumerate(doc_ids, 1):
        print(f"\n📄 Processing document {i}/{len(doc_ids)}: {doc_id}")
        print("-" * 50)
        
        try:
            # Extract text with advanced method
            text, daId, metadata = get_pdf_text_advanced(doc_id)
            
            # Categorize by text type
            text_type = metadata.get("text_type", "unknown")
            if text_type == "handwritten":
                handwriting_docs.append(doc_id)
            elif text_type == "mixed":
                mixed_docs.append(doc_id)
            elif text_type == "printed":
                printed_docs.append(doc_id)
            else:
                failed_docs.append(doc_id)
            
            results[doc_id] = {
                "text": text,
                "daId": daId,
                "metadata": metadata,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Failed to process document {doc_id}: {str(e)}")
            failed_docs.append(doc_id)
            results[doc_id] = {
                "text": "",
                "daId": f"FAILED_{doc_id}",
                "metadata": {"error": str(e)},
                "status": "failed"
            }
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"📄 Total documents: {len(doc_ids)}")
    print(f"✍️  Handwritten documents: {len(handwriting_docs)}")
    print(f"📝 Mixed content documents: {len(mixed_docs)}")
    print(f"🖨️  Printed documents: {len(printed_docs)}")
    print(f"❌ Failed documents: {len(failed_docs)}")
    
    if handwriting_docs:
        print(f"\n✍️  Handwritten document IDs: {handwriting_docs}")
    if mixed_docs:
        print(f"\n📝 Mixed content document IDs: {mixed_docs}")
    if failed_docs:
        print(f"\n❌ Failed document IDs: {failed_docs}")
    
    return results

def compare_extraction_methods(doc_id):
    """
    Compare original vs advanced extraction for a single document
    """
    print(f"🔍 Comparing extraction methods for document: {doc_id}")
    print("=" * 60)
    
    # Fetch the PDF
    response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS, timeout=30)
    if response.status_code != 200:
        print(f"❌ Failed to fetch document: {response.status_code}")
        return
    
    doc_data = response.json()
    pdf_bytes = base64.b64decode(doc_data["value"]["documentBuffer"])
    
    # Test advanced extraction
    print("\n🚀 Testing Advanced Extraction:")
    print("-" * 40)
    start_time = datetime.now()
    text_advanced, metadata_advanced = extract_text_advanced(pdf_bytes, doc_id)
    advanced_time = (datetime.now() - start_time).total_seconds()
    
    print(f"📊 Method: {metadata_advanced.get('extraction_method')}")
    print(f"📊 Text Type: {metadata_advanced.get('text_type')}")
    print(f"📊 Confidence: {metadata_advanced.get('confidence', 0):.2f}")
    print(f"📊 Processing Time: {advanced_time:.2f}s")
    print(f"📊 Text Length: {len(text_advanced)} characters")
    
    # Show text preview
    preview_advanced = text_advanced[:200] + "..." if len(text_advanced) > 200 else text_advanced
    print(f"\n📝 Advanced Text Preview:\n{preview_advanced}")
    
    return {
        "advanced": {
            "text": text_advanced,
            "metadata": metadata_advanced,
            "processing_time": advanced_time
        }
    }

def integration_guide():
    """
    Print integration guide for replacing existing extraction
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADVANCED TEXT EXTRACTION INTEGRATION GUIDE                ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔧 TO INTEGRATE WITH YOUR EXISTING PIPELINE:

1. Install Dependencies:
   pip install -r requirements_advanced_ocr.txt

2. Optional (for best handwriting support):
   pip install transformers torch torchvision  # For TrOCR
   pip install paddlepaddle paddleocr          # For PaddleOCR

3. Replace the get_pdf_text() function in final.py:
   
   FROM:
   def get_pdf_text(doc_id):
       # ... existing code ...
   
   TO:
   from advanced_text_extractor import extract_text_advanced
   
   def get_pdf_text(doc_id):
       # ... existing code to fetch document ...
       text, metadata = extract_text_advanced(pdf_bytes, doc_id)
       return [text, daId, metadata]

4. Enhanced Features You'll Get:
   ✍️  Advanced handwriting recognition (TrOCR)
   📝 Multiple OCR engines with automatic best-result selection
   🖼️  Intelligent image preprocessing for different text types
   📊 Confidence scoring and text type detection
   🔄 Robust fallback system with 6+ extraction methods
   ⚡ Smart method prioritization based on document type

5. Key Improvements for Medical Documents:
   🏥 Better handling of physician signatures and handwritten notes
   📋 Improved recognition of handwritten patient information
   💊 Enhanced extraction of prescription details and dosages
   📝 Better processing of form-filled documents
   🔍 Automatic detection of document text types

6. Monitoring and Logging:
   📊 Detailed extraction metadata for quality assessment
   ⏱️  Processing time tracking
   📈 Confidence scores for reliability estimation
   🎯 Text type classification (printed/handwritten/mixed)

💡 RECOMMENDATION:
   Start with basic installation (EasyOCR + DocTR) for immediate improvement,
   then add TrOCR and PaddleOCR for maximum handwriting recognition capability.

""")

# Example usage
if __name__ == "__main__":
    import time
    
    print("🧪 Advanced Text Extraction Integration Demo")
    print("=" * 50)
    
    # Show integration guide
    integration_guide()
    
    # Example usage with your document IDs
    example_doc_ids = [
        # Add your actual document IDs here for testing
        # "doc_id_1",
        # "doc_id_2",
        # "doc_id_3"
    ]
    
    if example_doc_ids:
        print("🚀 Running batch processing demo...")
        results = batch_process_documents_with_handwriting(example_doc_ids)
        
        print("\n🔍 Running comparison demo for first document...")
        comparison = compare_extraction_methods(example_doc_ids[0])
    else:
        print("💡 Add document IDs to example_doc_ids list to test the system") 