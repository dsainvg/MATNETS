import re

with open("docs/concepts.md", "r") as f:
    content = f.read()

# Fix indented math blocks
content = content.replace("  $$", "$$")

with open("docs/concepts.md", "w") as f:
    f.write(content)
