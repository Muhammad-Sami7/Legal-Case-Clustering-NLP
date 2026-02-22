
import os
import re
import fitz  # PyMuPDF
import shutil
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from fpdf import FPDF
from rake_nltk import Rake

# ====== Folder Setup ======
INPUT_PDF_FOLDER = r"C:\Users\User\Desktop\jugdement_pdf"
CLEANED_TEXT_FOLDER = r"C:\Users\User\Desktop\Extracted_texts"
CLUSTERED_TEXT_FOLDER = r"C:\Users\User\Desktop\cluster_text"
CLUSTERED_PDF_FOLDER = r"C:\Users\User\Desktop\cluster_pdf"
MODEL_FOLDER = r"C:\Users\User\Desktop\Model1"
SUMMARY_FILE = os.path.join(MODEL_FOLDER, "cluster_summary.xlsx")
N_CLUSTERS = 5

for folder in [CLEANED_TEXT_FOLDER, CLUSTERED_TEXT_FOLDER, CLUSTERED_PDF_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ====== Step 1: PDF to Cleaned Text ======
def clean_text(text):
    text = re.sub(r'\n\d+\s*\n', '\n', text)  # Remove page numbers
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def convert_pdf_to_text():
    for file in os.listdir(INPUT_PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_PDF_FOLDER, file)
            text_path = os.path.join(CLEANED_TEXT_FOLDER, file.replace(".pdf", ".txt"))
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(clean_text(full_text))

# ====== Step 2: Vectorization + Clustering ======
def vectorize_and_cluster():
    documents = []
    filenames = []

    for file in os.listdir(CLEANED_TEXT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(CLEANED_TEXT_FOLDER, file)
            with open(path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
                filenames.append(file)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(documents)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)
    print(f"📈 Silhouette Score: {score:.4f}")

    joblib.dump(kmeans, os.path.join(MODEL_FOLDER, "kmeans.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_FOLDER, "vectorizer.pkl"))

    for cluster_id in range(N_CLUSTERS):
        os.makedirs(os.path.join(CLUSTERED_TEXT_FOLDER, f"cluster_{cluster_id}"), exist_ok=True)

    cluster_summary = []

    for fname, doc, label in zip(filenames, documents, labels):
        src = os.path.join(CLEANED_TEXT_FOLDER, fname)
        dst = os.path.join(CLUSTERED_TEXT_FOLDER, f"cluster_{label}", fname)
        shutil.copy(src, dst)

        rake = Rake()
        rake.extract_keywords_from_text(doc)
        keywords = ', '.join(rake.get_ranked_phrases()[:5])

        cluster_summary.append({"Filename": fname, "Cluster": label, "Top Keywords": keywords})

    df_summary = pd.DataFrame(cluster_summary)
    df_summary.to_excel(SUMMARY_FILE, index=False)

    return labels, filenames

# ====== Step 3: Convert Clustered Texts to PDFs ======
def text_to_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

def convert_clusters_to_pdf():
    for cluster_folder in os.listdir(CLUSTERED_TEXT_FOLDER):
        cluster_path = os.path.join(CLUSTERED_TEXT_FOLDER, cluster_folder)
        if os.path.isdir(cluster_path):
            pdf_cluster_folder = os.path.join(CLUSTERED_PDF_FOLDER, cluster_folder)
            os.makedirs(pdf_cluster_folder, exist_ok=True)
            for txt_file in os.listdir(cluster_path):
                txt_path = os.path.join(cluster_path, txt_file)
                pdf_name = txt_file.replace(".txt", ".pdf")
                pdf_path = os.path.join(pdf_cluster_folder, pdf_name)
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    text_to_pdf(text, pdf_path)

# ====== Master Pipeline ======
if __name__ == "__main__":
    print("🔄 Converting PDFs to cleaned text...")
    convert_pdf_to_text()
    print("🔄 Applying TF-IDF and KMeans clustering (k=5)...")
    labels, filenames = vectorize_and_cluster()
    print("🔄 Converting clustered text files back to PDFs...")
    convert_clusters_to_pdf()
    print("✅ All steps completed! Clustered PDFs saved in:", CLUSTERED_PDF_FOLDER)
    print("📊 Summary Excel saved at:", SUMMARY_FILE)
