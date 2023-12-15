from dotenv import load_dotenv
load_dotenv()


import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup

urls = [
    "https://policies.google.com/privacy?hl=en-US",
]


def scrape_web(urls):

    for url in urls:
        result = []
        req = requests.get(url)
        soup = BeautifulSoup(req.text, "html.parser")

        # keep only the heading tags up to h3, and p tags
        text = soup.find_all(["a", "h3", "p"])
        # text = soup.find_all(["a"])

        # remove the tags and keep the inner text
        text = [t.text for t in text]

        for i in text:
            try:
                result.append(i.encode('latin').decode("utf-8"))
            except:
                pass

        book_path = Path("web")
        if not book_path.exists():
            book_path.mkdir()

        # pagename = url.split("/")[-1]
        pagename = "data"

        with open(book_path / f"{pagename}.txt", "w") as f:
            f.write("\n".join(result))

scrape_web(urls)