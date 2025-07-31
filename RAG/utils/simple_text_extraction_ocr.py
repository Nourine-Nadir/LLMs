import os
import io
import pdfplumber
import pytesseract
from PIL import Image

# Optional: Configure Tesseract path (uncomment and set path if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_with_ocr(pdf_path, lang='fra', skip_pages=0, only_ocr=False):
    """
    Extract text from PDF using OCR for image-based pages or fallback if needed.
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        nb_pages = len(pdf.pages)
        for idx, page in enumerate(pdf.pages):
            print(f'Processing page {idx + 1}/{nb_pages}')
            if idx + 1 <= skip_pages:
                continue

            text = page.extract_text()

            if text and not only_ocr:
                print(f'Text found for {pdf_path}')
                full_text.append(text)
            else:
                print('Text not found --- Performing OCR...')
                try:
                    im = page.to_image(resolution=300).original

                    if isinstance(im, bytes):
                        im = Image.open(io.BytesIO(im))

                    text = pytesseract.image_to_string(im, lang=lang)
                    full_text.append(text)
                except Exception as e:
                    print(f"OCR failed on page {page.page_number}: {str(e)}")
                    full_text.append("")

    return "\n".join(filter(None, full_text))


def save_text_to_file(text, output_path):
    """
    Save extracted text to a file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully saved to {output_path}")


def process_pdfs_in_directory(input_dir, output_dir, lang='fra', only_ocr=True):
    """
    Process all PDFs in a directory and save extracted text files.
    """
    for filename in os.listdir(input_dir):
        # if not filename.lower().endswith(".pdf"):
        #     continue

        file_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)

        text = extract_text_with_ocr(file_path, lang=lang, only_ocr=only_ocr)
        save_text_to_file(text, output_path)


def main():
    # List of (input_dir, output_dir, language) tuples
    # jobs = [
    #     ("../Data/ENTTIC/ARABE/", "../Data/Text_data/enttic/", 'ara'),
    #     ("../Data/ENTTIC/Additional_Data/FRANCAIS/new/", "../Data/Text_data/enttic/", 'fra')
    # ]
    jobs = [
        ("../Data/law_documents_DZ/", "../Data/Text_data/algerian_laws/", 'fra'),

    ]

    for input_dir, output_dir, lang in jobs:
        print(f"\nProcessing PDFs in: {input_dir} | Language: {lang}")
        process_pdfs_in_directory(input_dir, output_dir, lang=lang)


if __name__ == "__main__":
    main()
