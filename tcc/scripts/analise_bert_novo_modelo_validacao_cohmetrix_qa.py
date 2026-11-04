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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    TEM_XGBOOST = True
except ImportError:
    TEM_XGBOOST = False
    print("AVISO: xgboost não instalado. Rode: pip install xgboost")

# Durante a execução final, é melhor NÃO esconder os avisos.
# Isso ajuda a identificar, por exemplo, problemas de convergência do MLP.
# warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.abspath(os.path.join(DIRETORIO_ATUAL, ".."))

# Preferência para a nova organização: dataset/anderson.
# Caso essa pasta não exista, mantém compatibilidade com a estrutura antiga: dataset/.
PASTA_DATASET_ANDERSON = os.path.join(DIRETORIO_RAIZ, "dataset", "anderson")
PASTA_DATASET_PADRAO = os.path.join(DIRETORIO_RAIZ, "dataset")
PASTA_DATASET = PASTA_DATASET_ANDERSON if os.path.isdir(PASTA_DATASET_ANDERSON) else PASTA_DATASET_PADRAO

LISTA_DE_DATASETS = ["df_11", "df_12"]
MODELO_SPACY = "pt_core_news_lg"
MODELO_EMBEDDINGS = "neuralmind/bert-base-portuguese-cased"

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
        raise KeyError(
            "Coluna 'question_text' não encontrada. Confira se o CSV de origem "
            "tem essa coluna (df_11_train.csv/df_12_train.csv já têm por padrão)."
        )

    df = df.copy()
    df["question_answer"] = (
        df["question_text"].fillna("").astype(str)
        + "\n Resposta: "
        + df["answer_text"].fillna("").astype(str)
    )
    return df


def gerar_variantes_preprocessamento(textos, nlp_pt, stop_words_pt, batch_size=64, n_process=1):
    textos = [str(t) if pd.notna(t) else "" for t in textos]

    saida = {
        "cru": [],
        "stopwords": [],
        "stopwords_lemma": [],
        "stopwords_lemma_pos": [],
    }

    for doc in nlp_pt.pipe(textos, batch_size=batch_size, n_process=n_process):
        tokens_cru, tokens_stop, tokens_lemma, tokens_lemma_pos = [], [], [], []

        for tok in doc:
            if not tok.is_alpha:
                continue

            tokens_cru.append(tok.text.lower())

            eh_stopword = tok.is_stop or tok.text.lower() in stop_words_pt
            if eh_stopword:
                continue

            tokens_stop.append(tok.text.lower())
            tokens_lemma.append(tok.lemma_.lower())

            if tok.pos_ in POS_RELEVANTES:
                tokens_lemma_pos.append(tok.lemma_.lower())

        saida["cru"].append(" ".join(tokens_cru))
        saida["stopwords"].append(" ".join(tokens_stop))
        saida["stopwords_lemma"].append(" ".join(tokens_lemma))
        saida["stopwords_lemma_pos"].append(" ".join(tokens_lemma_pos))

    return saida


def criar_indices_validacao(df_train, nome_dataset, random_state=42):
    """
    Cria uma divisão interna de treino/validação para selecionar modelos
    sem usar o conjunto de teste.

    - Em df_11, quando existir question_id, usa GroupShuffleSplit para separar
      perguntas entre treino interno e validação interna.
    - Nos demais casos, usa train_test_split estratificado pela nota, quando possível.
    """
    indices = np.arange(len(df_train))
    y = df_train["grade"].reset_index(drop=True)

    if nome_dataset == "df_11" and "question_id" in df_train.columns:
        grupos = df_train["question_id"].reset_index(drop=True)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        idx_treino, idx_validacao = next(splitter.split(indices, y, groups=grupos))
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


def carregar_e_juntar(nome_dataset, split, pasta=PASTA_DATASET):
    caminho = os.path.join(pasta, f"{nome_dataset}_{split}.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrei {caminho}")

    df = pd.read_csv(caminho).reset_index(drop=True)
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


def preparar_escala(X_train, X_test, idx_treino_int, idx_validacao, usar_escala):
    """
    Prepara matrizes para validação interna e teste final.

    Quando usar_escala=True, o scaler da validação é ajustado apenas no treino interno.
    Para o teste final, outro scaler é ajustado no treino completo e aplicado ao teste.
    """
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    if not usar_escala:
        return (
            X_train.iloc[idx_treino_int].reset_index(drop=True),
            X_train.iloc[idx_validacao].reset_index(drop=True),
            X_train,
            X_test,
        )

    scaler_val = StandardScaler()
    X_fit = pd.DataFrame(
        scaler_val.fit_transform(X_train.iloc[idx_treino_int]),
        columns=X_train.columns
    )
    X_val = pd.DataFrame(
        scaler_val.transform(X_train.iloc[idx_validacao]),
        columns=X_train.columns
    )

    scaler_final = StandardScaler()
    X_train_final = pd.DataFrame(
        scaler_final.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_final = pd.DataFrame(
        scaler_final.transform(X_test),
        columns=X_train.columns
    )

    return X_fit, X_val, X_train_final, X_test_final


def executar_experimento_asag(nome_dataset, nlp_pt, stop_words_pt, modelo_ia):
    print("\n" + "=" * 70)
    print(f"INICIANDO PIPELINE PARA: {nome_dataset} (texto = pergunta + resposta)")
    print("=" * 70)
    print(f"Pasta de dados utilizada: {PASTA_DATASET}")

    df_train, coh_train = carregar_e_juntar(nome_dataset, "train")
    df_test, coh_test = carregar_e_juntar(nome_dataset, "test")

    usar_cohmetrix = coh_train is not None and coh_test is not None
    if not usar_cohmetrix:
        print(" -> AVISO: Coh-Metrix não encontrado para este dataset. Rodando sem essas features.")

    textos_train = df_train["question_answer"].tolist()
    textos_test = df_test["question_answer"].tolist()
    y_train = df_train["grade"].reset_index(drop=True)
    y_test = df_test["grade"].reset_index(drop=True)

    idx_treino_int, idx_validacao = criar_indices_validacao(
        df_train=df_train,
        nome_dataset=nome_dataset,
        random_state=42
    )

    print(
        f" -> Validação interna: {len(idx_treino_int)} instâncias para treino interno "
        f"e {len(idx_validacao)} para validação."
    )

    # ---------- Embeddings AO VIVO (pergunta + resposta) ----------
    print(" -> Gerando embeddings (pergunta + resposta) — ao vivo, não usa mais o cache antigo...")
    emb_train = modelo_ia.encode(textos_train, show_progress_bar=False)
    emb_test = modelo_ia.encode(textos_test, show_progress_bar=False)
    cols_emb = [f"emb_{i}" for i in range(emb_train.shape[1])]
    df_emb_train = pd.DataFrame(emb_train, columns=cols_emb)
    df_emb_test = pd.DataFrame(emb_test, columns=cols_emb)

    # ---------- Variantes de pré-processamento para TF-IDF (pergunta + resposta) ----------
    print(" -> Gerando variantes de pré-processamento (stopwords / lemma / POS)...")
    variantes_train = gerar_variantes_preprocessamento(textos_train, nlp_pt, stop_words_pt)
    variantes_test = gerar_variantes_preprocessamento(textos_test, nlp_pt, stop_words_pt)

    nomes_variantes = {
        "cru": "Apenas TF-IDF (cru)",
        "stopwords": "Apenas TF-IDF (stopwords)",
        "stopwords_lemma": "Apenas TF-IDF (stopwords+lemma)",
        "stopwords_lemma_pos": "Apenas TF-IDF (stopwords+lemma+POS)",
    }

    print(" -> Vetorizando TF-IDF para cada variante...")
    tfidf_por_variante = {}
    for chave, rotulo in nomes_variantes.items():
        vec = TfidfVectorizer(max_features=1000)
        X_tr = pd.DataFrame(
            vec.fit_transform(variantes_train[chave]).toarray(),
            columns=vec.get_feature_names_out()
        )
        X_te = pd.DataFrame(
            vec.transform(variantes_test[chave]).toarray(),
            columns=vec.get_feature_names_out()
        )
        tfidf_por_variante[chave] = (X_tr, X_te)

    df_tfidf_train, df_tfidf_test = tfidf_por_variante["stopwords_lemma_pos"]

    df_emb_train = df_emb_train.reset_index(drop=True)
    df_emb_test = df_emb_test.reset_index(drop=True)
    if usar_cohmetrix:
        coh_train = coh_train.reset_index(drop=True)
        coh_test = coh_test.reset_index(drop=True)

    # ---------- Cenários ----------
    cenarios = {}
    for chave, rotulo in nomes_variantes.items():
        cenarios[rotulo] = tfidf_por_variante[chave]

    cenarios["Apenas Embeddings"] = (df_emb_train, df_emb_test)
    cenarios["TF-IDF (processado) + Embeddings"] = (
        pd.concat([df_tfidf_train, df_emb_train], axis=1),
        pd.concat([df_tfidf_test, df_emb_test], axis=1),
    )

    if usar_cohmetrix:
        cenarios["Apenas Coh-Metrix"] = (coh_train, coh_test)
        cenarios["TF-IDF (processado) + Coh-Metrix"] = (
            pd.concat([df_tfidf_train, coh_train], axis=1),
            pd.concat([df_tfidf_test, coh_test], axis=1),
        )
        cenarios["Coh-Metrix + Embeddings"] = (
            pd.concat([coh_train, df_emb_train], axis=1),
            pd.concat([coh_test, df_emb_test], axis=1),
        )
        cenarios["Tudo (TF-IDF processado + Coh-Metrix + Emb)"] = (
            pd.concat([df_tfidf_train, coh_train, df_emb_train], axis=1),
            pd.concat([df_tfidf_test, coh_test, df_emb_test], axis=1),
        )

    # ---------- Modelos ----------
    print(" -> Treinando e avaliando modelos...")
    modelos = {
        "Regressão Linear": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
        "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "Rede Neural (MLP)": MLPRegressor(random_state=42, max_iter=1000),
    }
    if TEM_XGBOOST:
        modelos["XGBoost"] = XGBRegressor(random_state=42)

    MODELOS_SEM_ESCALA = {"Árvore de Decisão", "Random Forest", "HistGradientBoosting", "XGBoost"}

    resultados = []
    melhor_r2_val = -float('inf')
    dados_melhor_modelo = {}

    for nome_cenario, (X_train_c, X_test_c) in cenarios.items():
        if X_train_c.shape[1] == 0:
            print(f"    * Cenário '{nome_cenario}' ficou sem colunas. Pulando.")
            continue

        X_train_c = X_train_c.reset_index(drop=True)
        X_test_c = X_test_c.reset_index(drop=True)

        for nome_modelo, modelo_base in modelos.items():
            usar_escala = nome_modelo not in MODELOS_SEM_ESCALA
            X_fit, X_val, X_train_final, X_test_final = preparar_escala(
                X_train_c,
                X_test_c,
                idx_treino_int,
                idx_validacao,
                usar_escala=usar_escala,
            )

            y_fit = y_train.iloc[idx_treino_int].reset_index(drop=True)
            y_val = y_train.iloc[idx_validacao].reset_index(drop=True)

            # Seleção por validação interna
            modelo_validacao = clone(modelo_base)
            modelo_validacao.fit(X_fit, y_fit)
            y_pred_val = modelo_validacao.predict(X_val)

            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2_val = r2_score(y_val, y_pred_val)

            # Avaliação final no teste: retreina com todo o treino e avalia no teste
            modelo_final = clone(modelo_base)
            modelo_final.fit(X_train_final, y_train)
            y_pred = modelo_final.predict(X_test_final)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            resultados.append({
                "Dataset": nome_dataset,
                "Cenário": nome_cenario,
                "Modelo": nome_modelo,
                "MAE_val": mae_val,
                "RMSE_val": rmse_val,
                "R²_val": r2_val,
                "MAE": mae,
                "RMSE": rmse,
                "R²": r2,
            })

            if r2_val > melhor_r2_val:
                melhor_r2_val = r2_val
                dados_melhor_modelo = {
                    "modelo": modelo_final,
                    "nome_modelo": nome_modelo,
                    "cenario": nome_cenario,
                    "r2_validacao": r2_val,
                    "r2_teste": r2,
                    "mae_teste": mae,
                    "rmse_teste": rmse,
                    "y_pred": y_pred,
                    "y_test": y_test,
                    "df_ref": df_test,
                }

    df_resultados = (
        pd.DataFrame(resultados)
        .sort_values(by="R²_val", ascending=False)
        .reset_index(drop=True)
    )
    df_resultados.to_csv(os.path.join(DIRETORIO_ATUAL, f"resultados_{nome_dataset}_bertimbau.csv"), index=False)

    print(f"\nConcluído! Resultados oficiais para {nome_dataset}:")
    print(df_resultados.head(10).round(4).to_string(index=False))
    if dados_melhor_modelo:
        print(
            f"Melhor Modelo pela validação: {dados_melhor_modelo['nome_modelo']} "
            f"({dados_melhor_modelo['cenario']}) | "
            f"R² validação: {dados_melhor_modelo['r2_validacao']:.4f} | "
            f"R² teste: {dados_melhor_modelo['r2_teste']:.4f}\n"
        )

    return df_resultados, dados_melhor_modelo


def tabela_ablacao_tfidf(df_resultados):
    variantes_ordem = [
        "Apenas TF-IDF (cru)",
        "Apenas TF-IDF (stopwords)",
        "Apenas TF-IDF (stopwords+lemma)",
        "Apenas TF-IDF (stopwords+lemma+POS)",
    ]
    df_abl = df_resultados[df_resultados["Cenário"].isin(variantes_ordem)].copy()
    df_abl["Cenário"] = pd.Categorical(df_abl["Cenário"], categories=variantes_ordem, ordered=True)
    tabela = df_abl.pivot_table(index="Modelo", columns="Cenário", values=["MAE", "RMSE", "R²"])
    return tabela.round(4)


def main():
    stop_words_pt, nlp_pt, modelo_ia = carregar_recursos()

    todas_as_tabelas = []
    melhores_modelos = {}

    print("Iniciando a bateria de experimentos...")

    for dataset in LISTA_DE_DATASETS:
        try:
            tabela_resultado, melhor_modelo = executar_experimento_asag(
                dataset, nlp_pt, stop_words_pt, modelo_ia
            )

            if tabela_resultado is not None and not tabela_resultado.empty:
                todas_as_tabelas.append(tabela_resultado)
                melhores_modelos[dataset] = melhor_modelo

                print(f"\n--- Ablação de pré-processamento TF-IDF: {dataset} ---")
                print(tabela_ablacao_tfidf(tabela_resultado).to_string())

        except Exception as e:
            print(f"Erro crítico ao processar {dataset}: {e}")
            import traceback
            traceback.print_exc()

    if todas_as_tabelas:
        relatorio_final = pd.concat(todas_as_tabelas, ignore_index=True)

        print("\n" + "=" * 70)
        print("RANKING GERAL DE TODOS OS EXPERIMENTOS (Top 20 pela validação)")
        print("=" * 70)
        print(relatorio_final.sort_values(by="R²_val", ascending=False).head(20).round(4).to_string(index=False))

        caminho_relatorio = os.path.join(DIRETORIO_ATUAL, "relatorio_geral_experimentos_asag_bertimbau.csv")
        relatorio_final.to_csv(caminho_relatorio, index=False)
        print(f"Relatório geral salvo em '{caminho_relatorio}'")
    else:
        print("\nNenhum resultado foi gerado. Verifique os logs acima para entender onde falhou.")


if __name__ == "__main__":
    main()
