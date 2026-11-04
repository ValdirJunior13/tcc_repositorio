"""
Verificação rápida (~3 min) do efeito da coluna 'indice_original' no df_11.

- Reproduz, com o MESMO carregamento/merge de analise_asag_mvp_validacao_cohmetrix_qa.py,
  os cenários "Apenas Coh-Metrix" e "TF-IDF (processado) + Coh-Metrix" do df_11,
  COM e SEM 'indice_original'.
- Não gera embeddings, não extrai Coh-Metrix, NÃO GRAVA NENHUM ARQUIVO: só imprime.
- Lê: df_11_train.csv, df_11_test.csv, cohmetrix_df_11_train.csv, cohmetrix_df_11_test.csv

Uso:  python verificar_indice_original_df11.py <pasta dataset/anderson>
Ex.:  python verificar_indice_original_df11.py ../dataset/anderson

Valores esperados na linha COM indice_original (resultados_df_11.csv):
  Apenas Coh-Metrix  XGBoost R2=-0.1007 | HistGradientBoosting R2=0.0182
  TF-IDF + Coh-Metrix HistGradientBoosting  MAE=0.5910 RMSE=0.7653 R2=0.2619
Se esses valores aparecerem, seu ambiente reproduz o TCC e a linha SEM é comparável.
"""
import os, re, sys, numpy as np, pandas as pd, spacy, nltk
from nltk.corpus import stopwords
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

P = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "dataset", "anderson")
DS = "df_11"
nltk.download("stopwords", quiet=True)
SW = set(stopwords.words("portuguese"))
NLP = spacy.load("pt_core_news_lg", disable=["ner", "parser"])
POS = {"NOUN", "VERB", "ADJ", "ADV"}
lim = lambda t: re.sub(r"\s+", " ", str(t).lower()).strip()

def carregar(split):
    df = pd.read_csv(os.path.join(P, f"{DS}_{split}.csv")).reset_index(drop=True)
    df["question_answer"] = df["question_text"].fillna("").astype(str) + "\n Resposta: " + df["answer_text"].fillna("").astype(str)
    df["chave_merge"] = df["question_answer"].apply(lim)
    c = pd.read_csv(os.path.join(P, f"cohmetrix_{DS}_{split}.csv"))
    c["chave_merge"] = c["resposta_original"].apply(lim)
    c = c.drop_duplicates(subset="chave_merge")
    m = df.merge(c, on="chave_merge", how="left")
    print(f"  {split}: registros associados ao Coh-Metrix = {m['resposta_original'].notna().mean():.1%}")
    rem = ["resposta_original", "nota_original", "answer_text", "grade", "chave_merge",
           "question_text", "question_answer", "id", "question_id"]
    X = m[[x for x in c.columns if x not in rem]].fillna(0)
    X.index = df.index
    return df, X

def tfidf_pos(textos):
    out = []
    for doc in NLP.pipe(textos, batch_size=64):
        toks = [t.lemma_.lower() for t in doc
                if t.is_alpha and not (t.is_stop or t.text.lower() in SW) and t.pos_ in POS]
        out.append(" ".join(toks))
    return out

def modelos():
    m = {"Regressão Linear": LinearRegression(), "KNN": KNeighborsRegressor(n_neighbors=5), "SVR": SVR(),
         "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
         "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
         "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
         "Rede Neural (MLP)": MLPRegressor(random_state=42, max_iter=1000)}
    if XGBRegressor is not None:
        m["XGBoost"] = XGBRegressor(random_state=42)
    return m
SEM_ESCALA = {"Árvore de Decisão", "Random Forest", "HistGradientBoosting", "XGBoost"}

def avaliar(nome, A, B, ytr, yte):
    A = A.reset_index(drop=True); B = B.reset_index(drop=True)
    for n, base in modelos().items():
        a, b = A, B
        if n not in SEM_ESCALA:
            s = StandardScaler(); a = pd.DataFrame(s.fit_transform(A), columns=A.columns); b = pd.DataFrame(s.transform(B), columns=A.columns)
        p = clone(base).fit(a, ytr).predict(b)
        print(f"{nome:44s} {n:22s} MAE={mean_absolute_error(yte,p):.4f} RMSE={np.sqrt(mean_squared_error(yte,p)):.4f} R2={r2_score(yte,p):.4f}", flush=True)

print("Carregando dados...")
dtr, Ctr = carregar("train"); dte, Cte = carregar("test")
ytr, yte = dtr["grade"].reset_index(drop=True), dte["grade"].reset_index(drop=True)
print("Gerando TF-IDF (stopwords+lemma+POS)...")
vec = TfidfVectorizer(max_features=1000)
Ttr = pd.DataFrame(vec.fit_transform(tfidf_pos(dtr["question_answer"].tolist())).toarray(), columns=vec.get_feature_names_out())
Tte = pd.DataFrame(vec.transform(tfidf_pos(dte["question_answer"].tolist())).toarray(), columns=vec.get_feature_names_out())

avaliar("Apenas TF-IDF (stopwords+lemma+POS)", Ttr, Tte, ytr, yte)
for rotulo, ctr, cte in [("COM indice_original", Ctr, Cte),
                         ("SEM indice_original", Ctr.drop(columns=["indice_original"]), Cte.drop(columns=["indice_original"]))]:
    print(f"\n=== {rotulo} ===")
    avaliar(f"Apenas Coh-Metrix [{rotulo}]", ctr, cte, ytr, yte)
    avaliar(f"TF-IDF + Coh-Metrix [{rotulo}]",
            pd.concat([Ttr, ctr.reset_index(drop=True)], axis=1),
            pd.concat([Tte, cte.reset_index(drop=True)], axis=1), ytr, yte)
