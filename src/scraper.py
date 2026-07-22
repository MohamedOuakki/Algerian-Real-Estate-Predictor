import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def create_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scrape_page(driver, page_number):
    url = f"https://www.ouedkniss.com/immobilier?p={page_number}"
    driver.get(url)

    try:
        # wait up to 15 seconds for at least one listing card to appear
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".announce-card"))
        )
    except Exception:
        print(f"  → Page {page_number}: timed out waiting for listings")
        return []

    listings = []
    cards = driver.find_elements(By.CSS_SELECTOR, ".announce-card")

    for card in cards:
        try:
            title    = card.find_element(By.CSS_SELECTOR, ".title").text.strip()
            price    = card.find_element(By.CSS_SELECTOR, ".price").text.strip()
            location = card.find_element(By.CSS_SELECTOR, ".location").text.strip()

            try:
                size = card.find_element(By.CSS_SELECTOR, ".size").text.strip()
            except Exception:
                size = None

            listings.append({
                "title":    title,
                "price":    price,
                "location": location,
                "size":     size
            })
        except Exception:
            continue

    return listings


def scrape_all(num_pages=10):
    driver = create_driver()
    all_listings = []

    try:
        for page in range(1, num_pages + 1):
            print(f"Scraping page {page}...")
            listings = scrape_page(driver, page)
            all_listings.extend(listings)
            print(f"  → Got {len(listings)} listings")
            time.sleep(3)
    finally:
        driver.quit()  # always close the browser, even if something crashes

    return all_listings

def debug_html(page_number=1):
    driver = create_driver()
    url = f"https://www.ouedkniss.com/immobilier?p={page_number}"
    driver.get(url)
    
    print("Waiting 10 seconds for page to fully load...")
    time.sleep(10)  # give JS plenty of time to render
    
    html = driver.page_source
    with open("data/raw/debug.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("HTML saved to data/raw/debug.html")
    print(f"Total HTML length: {len(html)} characters")
    driver.quit()

if __name__ == "__main__":
    #debug_html()
    data = scrape_all(num_pages=10)
    df = pd.DataFrame(data)
    df.to_csv("data/raw/listings.csv", index=False)
    print(f"\nDone. {len(df)} listings saved to data/raw/listings.csv")