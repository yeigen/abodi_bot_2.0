import os
import re

def clean_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.lower()
    
    content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
    
    content = content.replace("**", "")
    
    content = re.sub(r"^-{3,}\s*$", "", content, flags=re.MULTILINE)
    
    content = content.replace("artículo", "articulo")
    
    content = re.sub(r"^articulo \d+\s*$", "", content, flags=re.MULTILINE)
    
    content = re.sub(r"\[(http.*?)\]", r"\1", content)
    content = re.sub(r"(articulo \d+)o\.", r"\1.", content)

    start_marker = "url: http://www.secretariasenado.gov.co/senado/basedoc/constitucion_politica_1991.html#1"
    start_idx = content.find(start_marker)
    if start_idx != -1:
        content = content[start_idx:]
        
    content = re.sub(r"\n{3,}", "\n\n", content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Cleaned {filepath}")

def main():
    directory = "./markdowns"
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            clean_text_file(os.path.join(directory, filename))

if __name__ == "__main__":
    main()
