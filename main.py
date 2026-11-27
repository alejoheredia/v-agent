import asyncio
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from mistralai import agents

CV_PATH = Path(__file__).parent / "cv.md"
CV_TEXT = CV_PATH.read_text(encoding="utf-8")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())


@agents.tool
async def fetch_url(url: str) -> str:
    """Fetch a web page by URL and return its readable text content (truncated to 8000 chars)."""
    if not url.startswith(("http://", "https://")):
        raise agents.ToolError("url must start with http:// or https://")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; cv-agent/0.1)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(2_000_000).decode(charset, errors="replace")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:
        pass
    text = "\n".join(parser._chunks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if title_match:
        text = f"Title: {title_match.group(1).strip()}\n\n{text}"
    if len(text) > 8000:
        text = text[:8000] + "\n... [truncated]"
    return text or "(no text content found at this URL)"

INSTRUCTIONS = (
    "You are a CV assistant for Alejandro Heredia. Answer every question "
    "using only the CV reproduced below.\n\n"
    "Rules:\n"
    "- Be factual and concise; quote dates, employers, and metrics exactly as they appear.\n"
    "- If the answer is not in the CV, say you don't have that information instead of guessing.\n"
    "- When the user gives you a URL (or asks about a company/project the CV mentions and a URL is known), use the fetch_url tool to open the page, then combine what it says with the CV facts.\n"
    "- When useful, organize answers with short bullet points.\n\n"
    f"--- CV START ---\n{CV_TEXT}\n--- CV END ---"
)

agent = agents.Agent(
    model="mistral-large-4",
    name="cv-agent",
    instructions=INSTRUCTIONS,
    tools=[fetch_url],
)


async def main() -> None:
    result = await agent.run("What is Alejandro's most senior role and what does he do there?")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
