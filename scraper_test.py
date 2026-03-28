import time
import os
import subprocess
from playwright.sync_api import sync_playwright

def scrape_amazon_reviews(url, max_reviews=50):
    # 1. Install the browser on the Streamlit server if it's missing
    try:
        # We use a flag file to avoid installing it every single time someone clicks the button
        if not os.path.exists("/tmp/playwright_installed"):
            subprocess.run(["playwright", "install", "chromium"], check=True)
            with open("/tmp/playwright_installed", "w") as f:
                f.write("done")
    except Exception as e:
        print(f"Playwright install note: {e}")

    # 2. Start the Scraper
    with sync_playwright() as p:
        # Launch Browser (Stealthy mode)
        browser = p.chromium.launch(headless=True)
        # Set a real-looking User Agent to avoid being blocked by Amazon
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            # Go to Product Page
            page.goto(url, timeout=60000)
            time.sleep(2)

            # Click "See all reviews" to unlock more than the top 8-10
            all_reviews_link = page.query_selector("a[data-hook='see-all-reviews-link-foot']")
            if all_reviews_link:
                all_reviews_link.click()
                page.wait_for_load_state("networkidle")
        except Exception as e:
            print(f"Navigation error: {e}")

        reviews = []
        
        # 3. Loop through pages until we hit max_reviews
        while len(reviews) < max_reviews:
            # Find all review bodies
            elements = page.query_selector_all("[data-hook='review-body']")
            
            for el in elements:
                text = el.inner_text().strip()
                if text and text not in reviews:
                    reviews.append(text)
                if len(reviews) >= max_reviews:
                    break
            
            # Check for "Next Page" button
            next_button = page.query_selector("li.a-last a")
            if next_button and len(reviews) < max_reviews:
                try:
                    next_button.click()
                    page.wait_for_load_state("networkidle")
                    time.sleep(1.5) # Anti-bot delay
                except:
                    break
            else:
                break # No more pages available

        browser.close()
        return reviews[:max_reviews]
