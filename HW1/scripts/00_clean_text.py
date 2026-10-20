import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data/raw/pg2701.txt"
OUT = HERE.parent / "data/clean/moby_dick.txt"

text = RAW.read_text(encoding="utf-8")

text = re.sub(r"\*\*\* START OF.*?\*\*\*", "", text, count=1, flags=re.S)
text = re.sub(r"\*\*\* END OF.*", "", text, flags=re.S)

# Buscar la famosísima línea del inicio.
i = text.find("Call me Ishmael")
j = text.rfind("CHAPTER 1", 0, i) if i >= 0 else -1
if j >= 0:
    text = text[j:]

text = re.sub(r"\[Illustration[^\]]*\]", "", text)
text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(text, encoding="utf-8")

print(f"{len(text):,} chars, {len(text.split()):,} words -> {OUT}")