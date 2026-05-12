import asyncio
import os
import signal
import subprocess
import time
from playwright.async_api import async_playwright

async def main():
    print("Starting mkdocs serve...")
    process = subprocess.Popen(
        ["mkdocs", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    # Wait for the server to start
    time.sleep(3)

    print("Taking screenshots...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Test Concepts index
        await page.goto("http://localhost:8000/concepts/matrix-neurons/")
        await page.wait_for_timeout(1000)
        await page.screenshot(path="mkdocs_concepts_split.png", full_page=True)

        # Test API index
        await page.goto("http://localhost:8000/api/core-layers/")
        await page.wait_for_timeout(1000)
        await page.screenshot(path="mkdocs_api_split.png", full_page=True)

        await browser.close()

    print("Stopping mkdocs serve...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.wait()
    print("Verification complete.")

if __name__ == "__main__":
    asyncio.run(main())
