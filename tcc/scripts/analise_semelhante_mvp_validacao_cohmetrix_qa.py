import os
import re
import warnings
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

try:
    from xgboost import XGBRegressor
    TEM_XGBOOST = True
except ImportError:
    TEM_XGBOOST = False
    print("AVISO: xgboost não instalado. Rode: pip install xgboost --break-system-packages")

# warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.abspath(os.path.join(DIRETORIO_ATUAL, ".."))
PASTA_DATASET_PADRAO = os.path.join(DIRETORIO_RAIZ, "dataset")
PASTA_DATASET_ANDERSON = os.path.join(DIRETORIO_RAIZ, "dataset", "anderson")
PASTA_DATASET = PASTA_DATASET_ANDERSON if os.path.exists(PASTA_DATASET_ANDERSON) else PASTA_DATASET_PADRAO

LISTA_DE_DATASETS = ["df_11", "df_12"]
MODELO_SPACY = "pt_core_news_lg"
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
NOME_ARQUIVO_SAIDA = os.path.join(DIRETORIO_ATUAL, "relatorio_geral_experimentos_pesquisa_semelhante.csv")

POS_RELEVANTES = {"NOUN", "VERB", "ADJ", "ADV"}


def carregar_recursos():
    print("Baixando lista de stopwords (nltk)...")
    nltk.download('stopwords', quiet=True)
    stop_words_pt = set(stopwords.words('portuguese'))

    print(f"Carregando modelo spaCy '{MODELO_SPACY}' para lematização/POS...")
    nlp_pt = spacy.load(MODELO_SPACY, disable=["ner", "parser"])

    print(f"Carregando modelo de embeddings '{MODELO_EMBEDDINGS}'...")
    modelo_ia = SentenceTransformer(MODELO_EMBEDDINGS)

    print("Recursos prontos!")
    return stop_words_pt, nlp_pt, modelo_ia


def limpar_para_merge(texto):
    return re.sub(r'\s+', ' ', str(texto).lower()).strip()


def construir_texto_combinado(df):
    if "question_text" not in df.columns:
        raise KeyError("Coluna 'question_text' não encontrada no CSV.")
    df = df.copy()
    df["question_answer"] = (
        df["question_text"].fillna("").astype(str)
        + "\n Resposta: "
        + df["answer_text"].fillna("").astype(str)
    )
    return df


def preprocessar_para_tfidf(textos, nlp_pt, stop_words_pt, batch_size=64, n_process=1):
    textos = [str(t) if pd.notna(t) else "" for t in textos]
    saida = []

    for doc in nlp_pt.pipe(textos, batch_size=batch_size, n_process=n_process):
        tokens = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            if tok.is_stop or tok.text.lower() in stop_words_pt:
                continue
            if tok.pos_ not in POS_RELEVANTES:
                continue
            tokens.append(tok.lemma_.lower())
        saida.append(" ".join(tokens))

    return saida


def carregar_e_juntar(nome_dataset, split, pasta=PASTA_DATASET):
    caminho = os.path.join(pasta, f"{nome_dataset}_{split}.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrei {caminho}")

    df = pd.read_csv(caminho)
    df = construir_texto_combinado(df)
    df['chave_merge'] = df['question_answer'].apply(limpar_para_merge)

    caminho_coh = os.path.join(pasta, f"cohmetrix_{nome_dataset}_{split}.csv")
    coh_features = None
    if os.path.exists(caminho_coh):
        coh_raw = pd.read_csv(caminho_coh)
        coh_raw['chave_merge'] = coh_raw['resposta_original'].apply(limpar_para_merge)
        coh_dedup = coh_raw.drop_duplicates(subset='chave_merge')

        df_coh = df.merge(coh_dedup, on='chave_merge', how='left')
        colunas_remover = [
            'resposta_original', 'nota_original', 'answer_text', 'grade',
            'chave_merge', 'question_text', 'question_answer', 'id', 'question_id',
        ]
        colunas_features = [c for c in coh_dedup.columns if c not in colunas_remover]
        coh_features = df_coh[colunas_features].fillna(0)
        coh_features.index = df.index

    return df, coh_features



def criar_indices_validacao(df_train, nome_dataset, random_state=42):
    """
    Cria uma divisão interna de treino/validação para selecionar/ordenar
    os resultados sem usar o conjunto de teste.

    Para df_11, quando há question_id, separa grupos de questões na validação,
    aproximando o cenário de generalização para perguntas não vistas.

    Para os demais casos, usa divisão estratificada pelas notas, quando possível.
    """
    indices = np.arange(len(df_train))
    y = df_train["grade"].reset_index(drop=True)

    if nome_dataset == "df_11" and "question_id" in df_train.columns:
        grupos = df_train["question_id"].reset_index(drop=True)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=random_state
        )
        idx_treino, idx_validacao = next(
            splitter.split(indices, y, groups=grupos)
        )
        return idx_treino, idx_validacao

    contagem_classes = y.value_counts()
    stratify = y if len(contagem_classes) > 1 and contagem_classes.min() >= 2 else None

    idx_treino, idx_validacao = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=stratify
    )
    return idx_treino, idx_validacao

def executar_experimento_pesquisa_semelhante(nome_dataset, nlp_pt, stop_words_pt, modelo_ia):
    print("\n" + "=" * 70)
    print(f"INICIANDO PIPELINE (estilo artigo) PARA: {nome_dataset} (pergunta + resposta)")
    print("=" * 70)

    df_train, coh_train = carregar_e_juntar(nome_dataset, "train")
    df_test, coh_test = carregar_e_juntar(nome_dataset, "test")

    usar_cohmetrix = coh_train is not None and coh_test is not None

    textos_train = df_train["question_answer"].tolist()
    textos_test = df_test["question_answer"].tolist()
    y_train = df_train["grade"].reset_index(drop=True)
    y_test = df_test["grade"].reset_index(drop=True)

    idx_treino_int, idx_validacao = criar_indices_validacao(
        df_train=df_train,
        nome_dataset=nome_dataset,
        random_state=42
    )

    print(" -> Gerando embeddings (pergunta + resposta) — ao vivo...")
    emb_train = modelo_ia.encode(textos_train, show_progress_bar=False)
    emb_test = modelo_ia.encode(textos_test, show_progress_bar=False)
    cols_emb = [f"emb_{i}" for i in range(emb_train.shape[1])]
    df_emb_train = pd.DataFrame(emb_train, columns=cols_emb)
    df_emb_test = pd.DataFrame(emb_test, columns=cols_emb)

    print(" -> Pré-processando texto para TF-IDF (stopwords + lemma + POS)...")
    texto_proc_train = preprocessar_para_tfidf(textos_train, nlp_pt, stop_words_pt)
    texto_proc_test = preprocessar_para_tfidf(textos_test, nlp_pt, stop_words_pt)

    vec = TfidfVectorizer(max_features=1000)
    df_tfidf_train = pd.DataFrame(vec.fit_transform(texto_proc_train).toarray(),
                                   columns=vec.get_feature_names_out())
    df_tfidf_test = pd.DataFrame(vec.transform(texto_proc_test).toarray(),
                                  columns=vec.get_feature_names_out())

    df_emb_train = df_emb_train.reset_index(drop=True)
    df_emb_test = df_emb_test.reset_index(drop=True)
    if usar_cohmetrix:
        coh_train = coh_train.reset_index(drop=True)
        coh_test = coh_test.reset_index(drop=True)

    representacoes = {
        "TFIDF": (df_tfidf_train, df_tfidf_test),
        "Embeddings": (df_emb_train, df_emb_test),
    }
    if usar_cohmetrix:
        representacoes["CohMetrix"] = (coh_train, coh_test)
    else:
        print(" -> AVISO: Coh-Metrix não encontrado para este dataset. Rodando sem essa representação.")

    modelos = {
        "SVM": SVR(),
        "RF": RandomForestRegressor(n_estimators=100, random_state=42),
        "DT": DecisionTreeRegressor(random_state=42),
    }
    if TEM_XGBOOST:
        modelos["XGB"] = XGBRegressor(random_state=42)
    else:
        print(" -> AVISO: XGBoost indisponível, essa linha ficará faltando (assim como no artigo).")

    MODELOS_SEM_ESCALA = {"RF", "DT", "XGB"}

    resultados = []

    for nome_repr, (X_train_r, X_test_r) in representacoes.items():
        if X_train_r.shape[1] == 0:
            print(f"    * Representação '{nome_repr}' ficou sem colunas. Pulando.")
            continue

        colunas = X_train_r.columns

        scaler = StandardScaler()
        X_tr_escalado = pd.DataFrame(scaler.fit_transform(X_train_r), columns=colunas)
        X_te_escalado = pd.DataFrame(scaler.transform(X_test_r), columns=colunas)

        X_tr_cru = X_train_r.reset_index(drop=True)
        X_te_cru = X_test_r.reset_index(drop=True)

        for nome_modelo, modelo_base in modelos.items():
            if nome_modelo in MODELOS_SEM_ESCALA:
                X_tr_sc, X_te_sc = X_tr_cru, X_te_cru
            else:
                X_tr_sc, X_te_sc = X_tr_escalado, X_te_escalado

            # ---------- Seleção/controle por validação interna ----------
            X_fit = X_tr_sc.iloc[idx_treino_int].reset_index(drop=True)
            X_val = X_tr_sc.iloc[idx_validacao].reset_index(drop=True)
            y_fit = y_train.iloc[idx_treino_int].reset_index(drop=True)
            y_val = y_train.iloc[idx_validacao].reset_index(drop=True)

            modelo_validacao = clone(modelo_base)
            modelo_validacao.fit(X_fit, y_fit)
            y_pred_val = modelo_validacao.predict(X_val)

            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2_val = r2_score(y_val, y_pred_val)

            # ---------- Avaliação final no teste ----------
            # O modelo final é treinado novamente com todo o conjunto de treino
            # e avaliado uma única vez no conjunto de teste.
            modelo_final = clone(modelo_base)
            modelo_final.fit(X_tr_sc, y_train)
            y_pred = modelo_final.predict(X_te_sc)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            resultados.append({
                "Dataset": nome_dataset,
                "Features": nome_repr,
                "Model": nome_modelo,
                "MAE_val": round(mae_val, 4),
                "RMSE_val": round(rmse_val, 4),
                "R2_val": round(r2_val, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            })

    # A ordenação usa MAE_val porque este experimento restrito é apresentado
    # no formato comparativo com MAE/RMSE. As métricas finais reportadas
    # continuam sendo MAE, RMSE e R2 do conjunto de teste.
    df_resultados = pd.DataFrame(resultados).sort_values(by="MAE_val").reset_index(drop=True)

    print(f"\nResultados (estilo artigo) para {nome_dataset}:")
    print(df_resultados.to_string(index=False))

    return df_resultados


def main():
    stop_words_pt, nlp_pt, modelo_ia = carregar_recursos()

    todas_as_tabelas = []

    for dataset in LISTA_DE_DATASETS:
        try:
            df_resultado = executar_experimento_pesquisa_semelhante(
                dataset, nlp_pt, stop_words_pt, modelo_ia)
            if df_resultado is not None and not df_resultado.empty:
                todas_as_tabelas.append(df_resultado)
        except Exception as e:
            print(f"Erro crítico ao processar {dataset}: {e}")
            import traceback
            traceback.print_exc()

    if todas_as_tabelas:
        relatorio_final = pd.concat(todas_as_tabelas, ignore_index=True)

        print("\n" + "=" * 70)
        print("TABELA FINAL - ESTILO Table 5/6 DO ARTIGO (Features/Model x MAE/RMSE)")
        print("=" * 70)

        tabela_pivot = relatorio_final.pivot_table(
            index=["Features", "Model"],
            columns="Dataset",
            values=["MAE", "RMSE"]
        )
        print(tabela_pivot.to_string())

        relatorio_final.to_csv(NOME_ARQUIVO_SAIDA, index=False)
        print(f"\nRelatório salvo como '{NOME_ARQUIVO_SAIDA}'")
    else:
        print("\nNenhum resultado foi gerado. Verifique os logs acima para entender onde falhou.")


if __name__ == "__main__":
    main()