from playwright.sync_api import sync_playwright

def scrape_amazon_reviews(url):
    with sync_playwright() as p:
        # Launching with headless=False so you can visually see if Amazon throws a CAPTCHA
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print("Navigating to product page...")
        page.goto(url)

        print("Waiting for the review section to load...")
        try:
            # Amazon frequently uses this specific data-hook for review text
            page.wait_for_selector('[data-hook="review-body"]', timeout=8000)

            # Grab all the review elements currently loaded on the page
            reviews = page.locator('[data-hook="review-body"]').all_inner_texts()

            print(f"\nSuccess! Extracted {len(reviews)} reviews.")
            
            # Print the first 3 reviews (truncated) to verify
            for i, review in enumerate(reviews[:3]):
                print(f"\n--- Review {i+1} ---")
                print(review.strip()[:200] + "...\n")

        except Exception as e:
            print("\nExtraction failed. We either hit a CAPTCHA or Amazon changed their HTML structure.")
            print(f"Error details: {e}")

        browser.close()

# Using a standard product link to test
test_url = "https://www.amazon.in/Apple-iPhone-13-128GB-Midnight/dp/B09G9HD6PD/"
scrape_amazon_reviews(test_url)
