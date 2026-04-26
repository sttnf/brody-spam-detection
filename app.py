
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import torch
import nltk
import re
import gensim
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- INITIALIZATION ---
# Fixed: Downloading punkt and punkt_tab to resolve NLTK errors
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"NLTK Warning: {e}")

# Load Local Models
print("⏳ Loading Ensemble Models...")
trained_models = joblib.load('trained_models.pkl')
tfidf_vec = joblib.load('tfidf_vec.pkl')
w2v = joblib.load('w2v_model.pkl')

# Load YOUR Fine-Tuned IndoBERT (Ensures weights are NOT missing)
BERT_MODEL_NAME = "rafaalrazzak/judol-detector-indobert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⏳ Loading Fine-Tuned IndoBERT from {BERT_MODEL_NAME}...")

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()

# --- HELPER FUNCTIONS ---

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def doc_to_vec(tokens, model):
    if hasattr(model, 'wv'):
        vecs = [model.wv[word] for word in tokens if word in model.wv]
    else:
        vecs = [model[word] for word in tokens if word in model]
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    return np.zeros(getattr(model, 'vector_size', 100))

def bert_predict_batch(texts, batch_size=16):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        encoding = bert_tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            logits = bert_model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)

# --- PREDICTION ENGINE ---

def predict_ensemble(komentar, model_view):
    if not komentar.strip():
        return "⚠️ Masukkan teks terlebih dahulu.", None
    
    clean = preprocess_text(komentar)
    rows = []
    
    # 1. TF-IDF Models
    tfidf_feat = tfidf_vec.transform([clean])
    tfidf_list = {
        'Logistic Regression': 'Logistic Regression (TF-IDF)',
        'Complement NB': 'Complement NB (TF-IDF)',
        'Linear SVM': 'Linear SVM (TF-IDF)',
        'Random Forest': 'Random Forest (TF-IDF)',
        'XGBoost': 'XGBoost (TF-IDF)'
    }
    
    for label, key in tfidf_list.items():
        m = trained_models.get(key)
        if m:
            p = m.predict(tfidf_feat)[0]
            prob = m.predict_proba(tfidf_feat)[0][1] if hasattr(m, 'predict_proba') else float(p)
            rows.append({'Model': label, 'Fitur': 'TF-IDF', 'Prediksi': '🚫 SPAM' if p else '✅ HAM', 'Score': f'{prob:.2%}'})

    # 2. Word2Vec Models
    tokens = word_tokenize(clean)
    w2v_feat = doc_to_vec(tokens, w2v).reshape(1, -1)
    w2v_list = {'Logistic Regression (W2V)': 'Logistic Regression (W2V)', 'LightGBM (W2V)': 'LightGBM (W2V)'}
    
    for label, key in w2v_list.items():
        m = trained_models.get(key)
        if m:
            p = m.predict(w2v_feat)[0]
            prob = m.predict_proba(w2v_feat)[0][1] if hasattr(m, 'predict_proba') else float(p)
            rows.append({'Model': label, 'Fitur': 'Word2Vec', 'Prediksi': '🚫 SPAM' if p else '✅ HAM', 'Score': f'{prob:.2%}'})

    # 3. IndoBERT
    bert_prob = bert_predict_batch([komentar])[0]
    rows.append({'Model': 'IndoBERT Fine-Tuned', 'Fitur': 'Transformer', 'Prediksi': '🚫 SPAM' if bert_prob >= 0.5 else '✅ HAM', 'Score': f'{bert_prob:.2%}'})

    df_res = pd.DataFrame(rows)
    
    # Filter view
    if model_view != "Ensemble (All Models)":
        df_res = df_res[df_res['Model'].str.contains(model_view)]

    # Verdict logic
    spam_votes = sum(1 for r in rows if 'SPAM' in r['Prediksi'])
    total = len(rows)
    if spam_votes > total/2:
        verdict = f"### 🚫 HASIL: TERDETEKSI SPAM\n({spam_votes}/{total} model setuju)"
    else:
        verdict = f"### ✅ HASIL: KOMENTAR AMAN\n({total - spam_votes}/{total} model setuju)"
        
    return verdict, df_res

# --- UI DESIGN ---
with gr.Blocks(title="Windah Spam Shield") as demo:
    gr.Markdown("# 🛡️ Windah Basudara — Spam Shield")
    gr.Markdown("Sistem deteksi spam & judi online otomatis.")
    
    with gr.Tabs():
        with gr.Tab("🔍 Single Detection"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(lines=5, label="Input Komentar", placeholder="Tempel komentar di sini...")
                    model_picker = gr.Dropdown(
                        choices=["Ensemble (All Models)", "IndoBERT", "XGBoost", "Logistic Regression", "Random Forest"],
                        value="Ensemble (All Models)",
                        label="Pilih Viewport"
                    )
                    run_btn = gr.Button("🚀 Deteksi Sekarang", variant="primary")
                with gr.Column(scale=3):
                    out_verdict = gr.Markdown("Hasil analisa akan muncul di sini.")
                    out_df = gr.Dataframe(interactive=False)
            
            gr.Examples(
                examples=[
                    ["W bang windah emang ga ada lawan!"],
                    ["INFO LINK GACOR HARI INI CEK BIO"],
                    ["Bang main game horror lagi dong"],
                    ["Depo 10k dapet bonus 100% langsung wd"]
                ],
                inputs=input_text
            )

        with gr.Tab("📁 Batch Processing"):
            gr.Markdown("### 📊 Mass Detection (CSV)")
            file_in = gr.File(label="Upload CSV (Harus punya kolom 'text')")
            batch_btn = gr.Button("⚡ Jalankan Batch", variant="primary")
            file_out = gr.File(label="Download Hasil")

            def batch_proc(file):
                df = pd.read_csv(file.name)
                col = 'text' if 'text' in df.columns else df.columns[0]
                probs = bert_predict_batch(df[col].astype(str).tolist())
                df['Prediction'] = ['SPAM' if p >= 0.5 else 'HAM' for p in probs]
                df['Confidence'] = [f'{p:.2%}' for p in probs]
                out = "spam_results.csv"
                df.to_csv(out, index=False)
                return out

            batch_btn.click(batch_proc, inputs=[file_in], outputs=[file_out])

    run_btn.click(predict_ensemble, inputs=[input_text, model_picker], outputs=[out_verdict, out_df])

if __name__ == "__main__":
    # Theme moved to launch for Gradio 6.0 compatibility
    demo.launch(theme=gr.themes.Soft())
