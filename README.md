# Unsupervised Clustering of Legal Judgment Documents

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Project Overview
This project applies **unsupervised machine learning** to automatically cluster legal judgment PDF documents collected from the **Lahore High Court**.

Using **TF-IDF vectorization** and **K-Means clustering**, the system groups similar legal documents based on textual content. Additionally, **RAKE keyword extraction** is used to summarize document themes, and clustered files are reconstructed into organized PDF folders.

The objective is to assist legal professionals in efficiently organizing and analyzing large volumes of judgments.

---

## Dataset
- 50 legal judgment PDFs manually collected from Lahore High Court.
- Documents were processed without predefined labels (unsupervised learning).
- Raw PDFs are not included due to legal and licensing considerations.

---

## Objective
- Automatically cluster legal judgment PDFs by content similarity.  
- Extract meaningful keywords from each document.  
- Organize clustered documents into structured folders.  
- Reconstruct clustered text files back into PDFs for practical use.

---

## Methodology

### 1. PDF Text Extraction
- Used `PyMuPDF` to extract text from PDF files.
- Cleaned text (removed page numbers, special characters, extra whitespace).
- Saved cleaned content as `.txt` files.

### 2. TF-IDF Vectorization
- Converted text into numerical features using `TfidfVectorizer`.
- Parameters:
  - `stop_words='english'`
  - `max_df=0.9`
  - `min_df=2`

### 3. K-Means Clustering
- Clustered documents into **5 clusters** using:
  - `KMeans(n_clusters=5)`
- Saved trained model and vectorizer using `joblib`.
- Evaluated using **Silhouette Score**.

### 4. Keyword Extraction
- Used **RAKE** to extract top 5 keywords per document.
- Stored keywords with filenames and cluster IDs in `cluster_summary.xlsx`.

### 5. Cluster Organization
- Grouped `.txt` files into cluster-specific folders.
- Converted clustered `.txt` files back to PDFs using `FPDF`.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Number of Documents | 50 |
| Number of Clusters | 5 |
| Silhouette Score | 0.0359 |
| Model Directory | `Model1/` |
| Summary File | `cluster_summary.xlsx` |
| Clustered PDFs | `cluster_pdf/cluster_X/*.pdf` |

---

## Observations
- Clustering successfully grouped documents, but:
- **Low Silhouette Score (0.0359)** indicates overlapping textual patterns in legal judgments.
- Legal language is highly similar across categories, making unsupervised separation challenging.
- Keyword extraction helped interpret cluster themes.

---

