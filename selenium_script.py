from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print(f"Username: {os.getenv('DA_USERNAME')}")
print(f"Password: {'*' * len(os.getenv('DA_PASSWORD') or '')}")
print(f"Document ID: {os.getenv('DOCUMENT_ID')}")

class DoctorAllianceBot:
    def __init__(self):
        # Set up Chrome options
        self.chrome_options = Options()
        # self.chrome_options.add_argument('--headless')  # Uncomment to run in headless mode
        self.chrome_options.add_argument('--start-maximized')
        self.chrome_options.add_argument('--disable-notifications')
        
        # Initialize the driver
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.chrome_options
        )
        self.wait = WebDriverWait(self.driver, 20)

    def login(self):
        """Login to Doctor Alliance backoffice"""
        try:
            print("Navigating to login page...")
            # Navigate to the login page
            self.driver.get("https://backoffice.doctoralliance.com/")
            
            # Wait for page to load completely
            time.sleep(3)
            
            print("Looking for login elements...")
            # Try different selectors for username and password
            try:
                # First try by name
                username = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']"))
                )
                password = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
                )
            except TimeoutException:
                print("Trying alternative selectors...")
                # Try alternative selectors
                username = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Username']"))
                )
                password = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Password']"))
                )

            print("Found login elements, entering credentials...")
            # Clear any existing values
            username.clear()
            password.clear()
            
            # Get credentials from environment variables
            username_value = os.getenv("DA_USERNAME")
            password_value = os.getenv("DA_PASSWORD")
            
            print(f"Using username: {username_value}")
            
            # Send keys with explicit wait between characters
            for char in username_value:
                username.send_keys(char)
                time.sleep(0.1)
            
            for char in password_value:
                password.send_keys(char)
                time.sleep(0.1)
            
            print("Credentials entered, looking for login button...")
            
            # Try different selectors for the login button
            try:
                login_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
                )
            except TimeoutException:
                print("Trying alternative button selector...")
                login_button = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.login-button"))
                )
            
            print("Found login button, clicking...")
            # Use JavaScript click as a fallback
            self.driver.execute_script("arguments[0].click();", login_button)
            
            print("Login button clicked, proceeding without waiting for full dashboard...")
            return True
            
        except Exception as e:
            print(f"Login failed with error: {str(e)}")
            print("Current URL:", self.driver.current_url)
            print("Page source preview:", self.driver.page_source[:500])
            return False

    def navigate_to_search(self):
        """Navigate to the search section"""
        try:
            print("Waiting for sidebar to load...")
            time.sleep(2)  # Add a small delay for the sidebar to become interactive
            
            print("Looking for Search link...")
            # Try multiple selectors for the search link
            try:
                search_link = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Search')]"))
                )
            except TimeoutException:
                print("Trying alternative search link selector...")
                try:
                    search_link = self.wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='Search']"))
                    )
                except TimeoutException:
                    print("Trying even more alternatives...")
                    search_link = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'Search')]"))
                    )
            
            print("Found Search link, clicking...")
            # Use JavaScript click as a fallback
            self.driver.execute_script("arguments[0].click();", search_link)
            time.sleep(2)  # Wait for page transition
            
            return True
            
        except Exception as e:
            print(f"Navigation to Search failed: {str(e)}")
            print("Current URL:", self.driver.current_url)
            print("Page source preview:", self.driver.page_source[:500])
            self.driver.save_screenshot("search_error.png")
            return False

    def perform_search(self):
        """Perform search operation"""
        try:
            print("Waiting for search form elements...")
            # Wait for search text input
            search_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Enter search text...']"))
            )
            
            # Get document ID from environment variables
            document_id = os.getenv("DOCUMENT_ID")
            print(f"Entering document ID: {document_id}")
            
            # Clear and enter document ID
            search_input.clear()
            for char in document_id:
                search_input.send_keys(char)
                time.sleep(0.1)
            
            # Wait for and select from dropdown
            print("Looking for search type dropdown...")
            try:
                # Click the Select2 dropdown to open it
                dropdown = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".select2-selection"))
                )
                dropdown.click()
                time.sleep(1)  # Wait for dropdown to open
                
                # Now find and click the "Documents" option in the Select2 dropdown
                # The Select2 dropdown options are usually in a separate div at the bottom of the page
                documents_option = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//li[contains(@class, 'select2-results__option') and contains(text(), 'Documents')]"))
                )
                documents_option.click()
                time.sleep(1)  # Wait for selection to register
                
            except TimeoutException as e:
                print(f"Failed to interact with dropdown: {str(e)}")
                print("Taking screenshot of dropdown state...")
                self.driver.save_screenshot("dropdown_error.png")
                
                # Try alternative method using JavaScript
                print("Trying alternative method using JavaScript...")
                try:
                    # Set the value using JavaScript
                    self.driver.execute_script("""
                        var select = document.querySelector('select[name="SearchType"]');
                        if(select) {
                            select.value = 'Documents';
                            // Trigger change event for Select2
                            var event = new Event('change', { bubbles: true });
                            select.dispatchEvent(event);
                        }
                    """)
                    time.sleep(1)
                except Exception as js_error:
                    print(f"JavaScript fallback failed: {str(js_error)}")
                    raise
            
            print("Clicking search button...")
            # Find and click the search button
            try:
                search_button = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-primary"))
                )
            except TimeoutException:
                search_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Search')]"))
                )
            
            self.driver.execute_script("arguments[0].click();", search_button)
            
            print("Search initiated successfully!")
            return True
            
        except Exception as e:
            print(f"Search operation failed: {str(e)}")
            print("Current URL:", self.driver.current_url)
            print("Page source preview:", self.driver.page_source[:500])
            print("Taking screenshot of current state...")
            self.driver.save_screenshot("search_form_error.png")
            
            # Print the HTML of the dropdown for debugging
            try:
                dropdown_html = self.driver.find_element(By.CSS_SELECTOR, ".select2-container").get_attribute('outerHTML')
                print("Dropdown HTML:", dropdown_html)
            except:
                print("Could not get dropdown HTML")
            
            return False

    def wait_for_search_results(self):
        """Wait for search results to load"""
        try:
            print("Waiting for search results table to load...")
            # Wait for the table to be present
            table = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'table')]"))
            )
            
            # Wait for at least one row of data (excluding header)
            rows = self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, "//table[contains(@class, 'table')]//tbody//tr"))
            )
            
            print(f"Found {len(rows)} result rows")
            return True
            
        except TimeoutException:
            print("Search results did not load within timeout period")
            return False

    def extract_physician_npi(self, physician_text):
        """Extract NPI number from physician text (numbers within brackets)"""
        try:
            # Use regex to find numbers within brackets
            match = re.search(r'\[(\d+)\]', physician_text)
            if match:
                npi = match.group(1)
                print(f"Extracted NPI: {npi}")
                return npi
            else:
                print(f"No NPI found in: {physician_text}")
                return None
        except Exception as e:
            print(f"Error extracting NPI: {str(e)}")
            return None

    def extract_table_data_with_npi(self):
        """Extract table data and specifically extract NPI from physician field"""
        try:
            print("Extracting table data with NPI extraction...")
            
            # Find the table
            table = self.driver.find_element(By.XPATH, "//table[contains(@class, 'table')]")
            
            # Extract headers
            headers = []
            header_elements = table.find_elements(By.XPATH, ".//thead//th")
            for header in header_elements:
                headers.append(header.text.strip())
            
            print(f"Table headers: {headers}")
            
            # Extract data rows
            rows_data = []
            row_elements = table.find_elements(By.XPATH, ".//tbody//tr")
            
            for row in row_elements:
                row_data = {}
                cells = row.find_elements(By.XPATH, ".//td")
                
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        cell_text = cell.text.strip()
                        row_data[headers[i]] = cell_text
                        
                        # If this is the physician column, also extract NPI
                        if headers[i].lower() == 'physician':
                            npi = self.extract_physician_npi(cell_text)
                            row_data['Physician_NPI'] = npi
                
                rows_data.append(row_data)
            
            print(f"Extracted {len(rows_data)} rows of data:")
            for i, row in enumerate(rows_data):
                print(f"Row {i + 1}: {row}")
                if 'Physician_NPI' in row and row['Physician_NPI']:
                    print(f"  -> NPI: {row['Physician_NPI']}")
            
            return {
                'headers': headers,
                'rows': rows_data
            }
            
        except Exception as e:
            print(f"Failed to extract table data: {str(e)}")
            return None

    def click_on_document_row(self, document_id=None):
        """Click on a specific document row in the table"""
        try:
            print(f"Looking for document row with ID: {document_id or 'any'}")
            
            # Find all rows in the table
            rows = self.driver.find_elements(By.XPATH, "//table[contains(@class, 'table')]//tbody//tr")
            
            for row in rows:
                # Get the ID from the first column
                id_cell = row.find_element(By.XPATH, ".//td[1]")
                row_id = id_cell.text.strip()
                
                print(f"Found row with ID: {row_id}")
                
                # If no specific document_id is provided, click the first row
                # Otherwise, click the row that matches the document_id
                if document_id is None or row_id == str(document_id):
                    print(f"Clicking on row with ID: {row_id}")
                    
                    # Try clicking on the ID cell or the entire row
                    try:
                        self.driver.execute_script("arguments[0].click();", id_cell)
                    except:
                        # If clicking the cell fails, try clicking the row
                        self.driver.execute_script("arguments[0].click();", row)
                    
                    time.sleep(3)  # Wait for any page transition
                    return True
            
            print("No matching document row found")
            return False
            
        except Exception as e:
            print(f"Failed to click on document row: {str(e)}")
            return False

    def extract_physician_npi_from_document(self):
        """Extract only the NPI number in brackets from the document page"""
        try:
            print("Extracting physician NPI from document page...")
            
            # Wait a bit for the page to fully load
            time.sleep(3)
            
            # Method 1: Look for NPI in the visible document header area
            try:
                # Look for elements that might contain physician info with NPI
                all_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Physician') or contains(text(), 'Dr. Andrade')]")
                
                for element in all_elements:
                    # Get both text and inner HTML to catch all possible formats
                    element_text = element.text.strip()
                    element_html = element.get_attribute('innerHTML')
                    
                    print(f"Checking element text: {element_text}")
                    print(f"Checking element HTML: {element_html}")
                    
                    # Look for NPI pattern [numbers] in both text and HTML
                    for content in [element_text, element_html]:
                        npi_match = re.search(r'\[(\d{10})\]', content)  # NPI is typically 10 digits
                        if npi_match:
                            npi = npi_match.group(1)
                            print(f"Found NPI: {npi}")
                            return npi
                        
                        # Also try for any number in brackets (not just 10 digits)
                        npi_match = re.search(r'\[(\d+)\]', content)
                        if npi_match:
                            npi = npi_match.group(1)
                            print(f"Found potential NPI: {npi}")
                            return npi
                        
            except Exception as e:
                print(f"Method 1 failed: {str(e)}")
            
            # Method 2: Look in the entire page source for the NPI pattern
            try:
                page_source = self.driver.page_source
                
                # First, look for the specific physician name followed by NPI
                physician_npi_pattern = r'Dr\.\s*Andrade[^[]*\[(\d+)\]'
                match = re.search(physician_npi_pattern, page_source, re.IGNORECASE)
                if match:
                    npi = match.group(1)
                    print(f"Found NPI for Dr. Andrade: {npi}")
                    return npi
                
                # Look for any physician pattern with NPI
                patterns = [
                    r'Physician[^[]*\[(\d+)\]',
                    r'Dr\.[^[]*\[(\d+)\]',
                    r'Andrade[^[]*\[(\d+)\]'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, page_source, re.IGNORECASE)
                    if matches:
                        npi = matches[0]
                        print(f"Found NPI using pattern {pattern}: {npi}")
                        return npi
                        
            except Exception as e:
                print(f"Method 2 failed: {str(e)}")
            
            # Method 3: Look for NPI in JavaScript variables or functions
            try:
                page_source = self.driver.page_source
                
                # Look for NPI in JavaScript code (as seen in your log)
                js_npi_patterns = [
                    r'"Dr\.\s*Andrade[^"]*",\s*"[^"]*",\s*"[^"]*"[^}]*physicianId[^:]*:\s*["\']?(\d+)["\']?',
                    r'physicianId[^:]*:\s*["\']?(\d+)["\']?',
                    r'"Dr\.\s*Andrade[^"]*"[^,]*,[^,]*,[^,]*,[^}]*?(\d{10})'
                ]
                
                for pattern in js_npi_patterns:
                    matches = re.findall(pattern, page_source, re.IGNORECASE | re.DOTALL)
                    if matches:
                        npi = matches[0]
                        if len(npi) >= 8:  # NPI should be at least 8 digits
                            print(f"Found NPI in JavaScript: {npi}")
                            return npi
                            
            except Exception as e:
                print(f"Method 3 failed: {str(e)}")
            
            print("Could not extract NPI from document page")
            return None
            
        except Exception as e:
            print(f"Failed to extract NPI: {str(e)}")
            return None

    def get_all_visible_text(self):
        """Get all visible text from the current page for debugging"""
        try:
            body = self.driver.find_element(By.TAG_NAME, "body")
            all_text = body.text
            print(f"All visible text on page: {all_text}")
            return all_text
        except Exception as e:
            print(f"Failed to get all visible text: {str(e)}")
            return None

    def select_text_from_table(self):
        """Select and copy text from the table"""
        try:
            print("Selecting text from table...")
            
            # Find the table
            table = self.driver.find_element(By.XPATH, "//table[contains(@class, 'table')]")
            
            # Use JavaScript to select all text in the table
            self.driver.execute_script("""
                var table = arguments[0];
                var range = document.createRange();
                range.selectNodeContents(table);
                var selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
            """, table)
            
            print("Table text selected successfully")
            
            # You can also copy to clipboard if needed
            self.driver.execute_script("document.execCommand('copy');")
            print("Table text copied to clipboard")
            
            return True
            
        except Exception as e:
            print(f"Failed to select table text: {str(e)}")
            return False

    def get_selected_text(self):
        """Get the currently selected text"""
        try:
            selected_text = self.driver.execute_script("return window.getSelection().toString();")
            print(f"Selected text: {selected_text}")
            return selected_text
        except Exception as e:
            print(f"Failed to get selected text: {str(e)}")
            return None

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    bot = DoctorAllianceBot()
    
    try:
        # Login to the system
        if bot.login():
            # Navigate to search
            if bot.navigate_to_search():
                time.sleep(2)  # Wait for page to stabilize
                
                # Perform search
                if bot.perform_search():
                    # Wait for search results to load
                    if bot.wait_for_search_results():
                        # Extract table data with NPI extraction
                        table_data = bot.extract_table_data_with_npi()
                        
                        # Get NPI values for further processing if needed
                        if table_data and table_data['rows']:
                            for row in table_data['rows']:
                                if 'Physician_NPI' in row and row['Physician_NPI']:
                                    print(f"Found Physician NPI in table: {row['Physician_NPI']}")
                        
                        # Click on the specific document row
                        document_id = os.getenv("DOCUMENT_ID")
                        if bot.click_on_document_row(document_id):
                            print("Successfully clicked on document row, now extracting physician NPI...")
                            
                            # Extract only the NPI from the opened document
                            physician_npi = bot.extract_physician_npi_from_document()
                            
                            if physician_npi:
                                print("\n" + "="*30)
                                print("EXTRACTED PHYSICIAN NPI:")
                                print("="*30)
                                print(f"NPI: {physician_npi}")
                                print("="*30)
                                
                                # Save to file for easy access
                                with open('physician_npi.txt', 'w') as f:
                                    f.write(f"NPI: {physician_npi}\n")
                                print("NPI saved to physician_npi.txt")
                                
                            else:
                                print("Could not extract physician NPI")
                                # Optional: save page source for debugging
                                with open('debug_page_source.html', 'w', encoding='utf-8') as f:
                                    f.write(bot.driver.page_source)
                                print("Page source saved to debug_page_source.html for debugging")
                        
                        # Keep the browser open for a while to see the results
                        print("Keeping browser open for 30 seconds...")
                        time.sleep(30)
                    else:
                        print("Search results did not load properly")
                else:
                    print("Search operation failed")
            else:
                print("Navigation to search failed, taking screenshot...")
                bot.driver.save_screenshot("navigation_error.png")
        else:
            print("Login failed, taking screenshot...")
            bot.driver.save_screenshot("login_error.png")
    
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        bot.driver.save_screenshot("error.png")
    
    finally:
        bot.close()

if __name__ == "__main__":
    main()