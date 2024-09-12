import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def scrape_website(url):
    visited = set()
    to_visit = [url]
    content = {}
    count_pages =0
    
    while to_visit:
        if count_pages > 10:
            break
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        
        visited.add(current_url)
        
        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract all text from the page
            text = soup.get_text()
            content[current_url] = text
            
            # Find all links on the page
            for link in soup.find_all('a', href=True):
                link_url = urljoin(current_url, link['href'])
                if urlparse(link_url).netloc == urlparse(url).netloc:  # Stay within the same domain
                    to_visit.append(link_url)
            count_pages+=1
        
        except requests.RequestException:
            continue
        print("page_count == ",count_pages,url)
    
    return content
