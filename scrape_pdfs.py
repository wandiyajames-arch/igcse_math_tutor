import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Set the URL of the page hosting the PDFs
TARGET_URL = "https://example-educational-site.com/igcse/math-0580/" 
DOWNLOAD_DIR = "data/raw"

def download_pdf(url, save_path):
    """Downloads a single PDF file securely."""
    try:
        # Stream the download to avoid overloading memory with large files
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded: {os.path.basename(save_path)}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")

def scrape_pdfs_from_page(url, download_dir):
    """Scrapes a webpage for PDF links and downloads them."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"🔍 Scanning {url} for PDFs...")
    
    try:
        # Add a realistic User-Agent to avoid basic bot-blockers
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to access webpage: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all anchor tags with 'href' attributes ending in '.pdf'
    pdf_links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.lower().endswith('.pdf'):
            # Ensure the URL is absolute
            full_url = urljoin(url, href)
            pdf_links.append(full_url)

    # Remove duplicates
    pdf_links = list(set(pdf_links))
    print(f"📄 Found {len(pdf_links)} PDF(s). Starting download...\n")

    for pdf_url in pdf_links:
        # Extract a clean filename from the URL
        filename = os.path.basename(urlparse(pdf_url).path)
        save_path = os.path.join(download_dir, filename)
        
        # Skip if we already downloaded it
        if os.path.exists(save_path):
            print(f"⏭️ Skipping (already exists): {filename}")
            continue
            
        download_pdf(pdf_url, save_path)

if __name__ == "__main__":
    scrape_pdfs_from_page(TARGET_URL, DOWNLOAD_DIR)