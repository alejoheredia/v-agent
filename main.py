import asyncio
from pathlib import Path

from mistralai import agents

CV_PATH = Path(__file__).parent / "cv.md"
CV_TEXT = CV_PATH.read_text(encoding="utf-8")

INSTRUCTIONS = (
    "You are a CV assistant for Alejandro Heredia. Answer every question "
    "using only the CV reproduced below.\n\n"
    "Rules:\n"
    "- Be factual and concise; quote dates, employers, and metrics exactly as they appear.\n"
    "- If the answer is not in the CV, say you don't have that information instead of guessing.\n"
    "- When useful, organize answers with short bullet points.\n\n"
    f"--- CV START ---\n{CV_TEXT}\n--- CV END ---"
)

agent = agents.Agent(
    model="mistral-large-4",
    name="cv-agent",
    instructions=INSTRUCTIONS,
    connectors=[agents.connector("web_search")],
)


async def main() -> None:
    result = await agent.run("What is Alejandro's most senior role and what does he do there?")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
