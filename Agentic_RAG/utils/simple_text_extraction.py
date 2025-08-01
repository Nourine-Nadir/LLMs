import pdfplumber

file_path = "../Data/Reglement Interieur de ENSV.pdf"
output_file = "../Data/Text_data/enttic/reglement_interieur.txt"

with pdfplumber.open(file_path) as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Save text to a file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)
