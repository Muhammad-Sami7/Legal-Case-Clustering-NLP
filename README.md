# Unsupervised Clustering of Legal Judgment Documents

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Project Overview
This project automatically organizes and clusters legal judgment PDF documents using **unsupervised machine learning**. It leverages **TF-IDF vectorization** and **K-Means clustering** to group similar documents, with **RAKE** keyword extraction and reconstruction of clustered PDFs.  

The goal is to help legal professionals retrieve and analyze grouped documents efficiently.

---

## Objective
- Automatically cluster legal judgment PDFs by content similarity.  
- Extract top keywords from each document for better insight.  
- Reconstruct clustered documents back to PDFs for practical use.

---

## Methodology

1. **PDF Text Extraction**
   - Used `PyMuPDF` to extract text from PDFs.  
   - Cleaned text (removed page numbers, special characters, whitespace).  
   - Saved as `.txt` files.

2. **TF-IDF Vectorization**
   - Converted text into numerical features using `TfidfVectorizer`.  
   - Parameters:
     - `stop_words='english'`
     - `max_df=0.9`
     - `min_df=2`

3. **K-Means Clustering**
   - Clustered documents into 5 groups using `KMeans(n_clusters=5)`.  
   - Saved trained model and vectorizer with `joblib`.  
   - **Performance Metric:** Silhouette Score = 0.0359

4. **Keyword Extraction**
   - Used RAKE to extract top 5 keywords per document.  
   - Stored keywords with filenames and cluster IDs in `cluster_summary.xlsx`.

5. **Cluster-wise Organization**
   - Grouped `.txt` files into cluster folders.  
   - Converted clustered `.txt` files back to PDFs using `FPDF`.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Number of Clusters | 5 |
| Silhouette Score | 0.0359 |
| Model Saved To | `Model1/` |
| Summary Excel File | `cluster_summary.xlsx` |
| Clustered PDFs | `cluster_pdf/cluster_X/*.pdf` |

---

## Observations
- Clustering works, but low silhouette score indicates overlapping features in legal text.  
- Better preprocessing (lemmatization, entity removal) or topic modeling (LDA) can improve results.  
- Converting documents back to PDFs aids real-world legal workflows.

---

## Future Improvements
- **Better Vectorization:** Use Word2Vec, Doc2Vec, or BERT embeddings.  
- **Dimensionality Reduction:** Apply PCA or t-SNE to improve cluster separation.  
- **Advanced Clustering:** Use DBSCAN, Agglomerative, or HDBSCAN.  
- **Evaluation:** Manual validation with domain experts.  
- **UI/Interface:** Streamlit or Flask app to upload and view clustered documents.

---

