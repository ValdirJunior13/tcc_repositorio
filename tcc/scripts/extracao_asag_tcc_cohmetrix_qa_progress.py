import argparse
import os
import sys
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Caminhos
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

PASTA_DATASET_ANDERSON = os.path.join(ROOT_DIR, "dataset", "anderson")
PASTA_DATASET_ROOT = os.path.join(ROOT_DIR, "dataset")

if os.path.exists(PASTA_DATASET_ANDERSON):
    PASTA_DATASET = PASTA_DATASET_ANDERSON
else:
    PASTA_DATASET = PASTA_DATASET_ROOT

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# Configurações
# ============================================================

ARQUIVOS_PARA_PROCESSAR = [
    "df_11_train.csv",
    "df_11_test.csv",
    "df_12_train.csv",
    "df_12_test.csv",
]

COLUNA_QUESTAO = "question_text"
COLUNA_RESPOSTA = "answer_text"
COLUNA_NOTA = "grade"

EXTRATOR_GLOBAL = None


# ============================================================
# Coh-Metrix
# ============================================================

def importar_extrator():
    from aibox.nlp.features.portuguese.cohmetrix import CohMetrixExtractor
    return CohMetrixExtractor


def obter_extrator():
    """
    Cria o CohMetrixExtractor apenas uma vez por processo.
    Isso evita recriar o extrator em cada linha.
    """
    global EXTRATOR_GLOBAL

    if EXTRATOR_GLOBAL is None:
        CohMetrixExtractor = importar_extrator()
        EXTRATOR_GLOBAL = CohMetrixExtractor()

    return EXTRATOR_GLOBAL


# ============================================================
# Utilitários
# ============================================================

def construir_texto_qa(row):
    texto_questao = str(row[COLUNA_QUESTAO]) if pd.notna(row.get(COLUNA_QUESTAO)) else ""
    texto_resposta = str(row[COLUNA_RESPOSTA]) if pd.notna(row.get(COLUNA_RESPOSTA)) else ""

    return texto_questao + "\n Resposta: " + texto_resposta


def extrair_metricas_linha(row_dict):
    texto_combinado = row_dict["texto_combinado"]
    nota = row_dict["nota"]
    indice_original = row_dict["indice_original"]

    metricas = {
        "resposta_original": texto_combinado,
        "nota_original": nota,
        "indice_original": indice_original,
    }

    if "id" in row_dict:
        metricas["id"] = row_dict["id"]

    if "question_id" in row_dict:
        metricas["question_id"] = row_dict["question_id"]

    texto_limpo = str(texto_combinado).strip()

    if len(texto_limpo) < 2 or not any(c.isalpha() for c in texto_limpo):
        return metricas

    try:
        extrator = obter_extrator()
        resultado = extrator.extract(texto_combinado)
        metricas_extraidas = resultado.as_dict()
        metricas.update(metricas_extraidas)

    except Exception as e:
        metricas["erro_cohmetrix"] = str(e)

    return metricas


def formatar_tempo(segundos):
    if segundos < 60:
        return f"{segundos:.1f}s"

    minutos = segundos / 60

    if minutos < 60:
        return f"{minutos:.1f} min"

    horas = minutos / 60
    return f"{horas:.1f} h"


def preparar_linhas(df):
    linhas = []

    for idx, row in df.iterrows():
        item = {
            "indice_original": int(idx),
            "texto_combinado": construir_texto_qa(row),
            "nota": row[COLUNA_NOTA],
        }

        if "id" in df.columns:
            item["id"] = row["id"]

        if "question_id" in df.columns:
            item["question_id"] = row["question_id"]

        linhas.append(item)

    return linhas


def salvar_csv_atomico(df, caminho):
    """
    Salva primeiro em .tmp e depois substitui o arquivo final.
    Isso reduz risco de corromper checkpoint se o processo for interrompido.
    """
    caminho_tmp = caminho + ".tmp"
    df.to_csv(caminho_tmp, index=False)
    os.replace(caminho_tmp, caminho)


def carregar_checkpoint(caminho_checkpoint):
    if not os.path.exists(caminho_checkpoint):
        return pd.DataFrame()

    try:
        df_checkpoint = pd.read_csv(caminho_checkpoint)
        print(f"🔁 Checkpoint encontrado: {len(df_checkpoint)} linhas já processadas.")
        return df_checkpoint
    except Exception as e:
        print(f"⚠️ Não foi possível ler o checkpoint. Um novo será criado. Erro: {e}")
        return pd.DataFrame()


def nome_arquivo_saida(nome_entrada, limit=None):
    """
    Em modo teste, não sobrescreve o arquivo final real.
    Isso evita gerar um cohmetrix_df_*.csv com só algumas linhas.
    """
    if limit is not None:
        return f"TEST_limit{limit}_cohmetrix_{nome_entrada}"

    return f"cohmetrix_{nome_entrada}"


def validar_arquivo_final_existente(caminho_saida, caminho_entrada, nome_saida):
    """
    Verifica se um arquivo final já existente parece completo.
    Se estiver incompleto, permite continuar pelo checkpoint.
    """
    try:
        df_existente = pd.read_csv(caminho_saida)
        total_entrada = len(pd.read_csv(caminho_entrada))

        if len(df_existente) >= total_entrada:
            print(f"✅ Arquivo final já existe e parece completo. Ignorando: {nome_saida}")
            return True

        print(
            f"⚠️ Arquivo final existe, mas parece incompleto "
            f"({len(df_existente)}/{total_entrada} linhas). "
            f"Continuando pelo checkpoint..."
        )
        return False

    except Exception as e:
        print(f"⚠️ Não foi possível validar o arquivo final existente: {e}")
        print("Continuando pelo checkpoint...")
        return False


def processar_arquivo(
    nome_entrada,
    n_jobs=2,
    batch_size=20,
    save_every_batches=1,
    limit=None,
    overwrite=False,
    reset_checkpoint=False
):
    caminho_entrada = os.path.join(PASTA_DATASET, nome_entrada)
    nome_saida = nome_arquivo_saida(nome_entrada, limit=limit)
    caminho_saida = os.path.join(PASTA_DATASET, nome_saida)

    nome_checkpoint = f"CHECKPOINT_{nome_saida}"
    caminho_checkpoint = os.path.join(PASTA_DATASET, nome_checkpoint)

    print("\n" + "=" * 70)
    print(f"📂 Processando: {nome_entrada}")
    print(f"📄 Saída: {nome_saida}")
    print(f"💾 Checkpoint: {nome_checkpoint}")
    print("=" * 70)

    if not os.path.exists(caminho_entrada):
        print(f"⚠️ Arquivo não encontrado: {caminho_entrada}")
        return

    if reset_checkpoint and os.path.exists(caminho_checkpoint):
        os.remove(caminho_checkpoint)
        print("🧹 Checkpoint anterior removido.")

    if overwrite and os.path.exists(caminho_saida):
        os.remove(caminho_saida)
        print("🧹 Arquivo final anterior removido por --overwrite.")

    if os.path.exists(caminho_saida) and not overwrite and limit is None:
        arquivo_completo = validar_arquivo_final_existente(
            caminho_saida=caminho_saida,
            caminho_entrada=caminho_entrada,
            nome_saida=nome_saida,
        )
        if arquivo_completo:
            return

    df = pd.read_csv(caminho_entrada)

    colunas_necessarias = {COLUNA_QUESTAO, COLUNA_RESPOSTA, COLUNA_NOTA}
    colunas_faltantes = colunas_necessarias - set(df.columns)

    if colunas_faltantes:
        raise ValueError(
            f"O arquivo {nome_entrada} não possui as colunas necessárias: {colunas_faltantes}"
        )

    if limit is not None:
        df = df.head(limit).copy()
        print(f"⚠️ Modo teste: processando apenas {limit} linhas.")

    linhas = preparar_linhas(df)
    total = len(linhas)

    df_checkpoint = carregar_checkpoint(caminho_checkpoint)

    resultados_acumulados = []
    indices_processados = set()

    if not df_checkpoint.empty and "indice_original" in df_checkpoint.columns:
        resultados_acumulados = df_checkpoint.to_dict("records")
        indices_processados = set(df_checkpoint["indice_original"].astype(int).tolist())

    linhas_pendentes = [
        item for item in linhas
        if item["indice_original"] not in indices_processados
    ]

    print(f"Total de linhas no arquivo: {total}")
    print(f"Já processadas: {len(indices_processados)}")
    print(f"Pendentes: {len(linhas_pendentes)}")
    print(f"n_jobs={n_jobs} | batch_size={batch_size}")

    if not linhas_pendentes:
        print("✅ Nada pendente. Gerando arquivo final a partir do checkpoint.")
        df_final = pd.DataFrame(resultados_acumulados).sort_values("indice_original")
        salvar_csv_atomico(df_final, caminho_saida)

        if limit is None and os.path.exists(caminho_checkpoint):
            os.remove(caminho_checkpoint)

        return

    inicio_arquivo = time.time()
    batches_processados = 0
    quantidade_inicial = len(resultados_acumulados)

    total_lotes = (len(linhas_pendentes) + batch_size - 1) // batch_size

    # Mantém o pool paralelo aberto durante o arquivo inteiro.
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for inicio in range(0, len(linhas_pendentes), batch_size):
            fim = min(inicio + batch_size, len(linhas_pendentes))
            lote = linhas_pendentes[inicio:fim]

            numero_lote = (inicio // batch_size) + 1

            print(
                f"\n➡️ Lote {numero_lote}/{total_lotes} "
                f"| pendentes {inicio + 1}-{fim} de {len(linhas_pendentes)}"
            )

            inicio_lote = time.time()

            resultados_lote = parallel(
                delayed(extrair_metricas_linha)(item)
                for item in lote
            )

            resultados_lote = [r for r in resultados_lote if r is not None]
            resultados_acumulados.extend(resultados_lote)

            batches_processados += 1

            tempo_lote = time.time() - inicio_lote
            total_processado = len(resultados_acumulados)

            tempo_total = time.time() - inicio_arquivo
            novos_processados = max(total_processado - quantidade_inicial, 1)

            media_por_linha = tempo_total / novos_processados
            restante = total - total_processado
            estimativa_restante = restante * media_por_linha

            erros = sum(
                1 for r in resultados_lote
                if "erro_cohmetrix" in r and pd.notna(r["erro_cohmetrix"])
            )

            print(
                f"✅ Lote concluído em {formatar_tempo(tempo_lote)} | "
                f"Total: {total_processado}/{total} "
                f"({100 * total_processado / total:.2f}%) | "
                f"Média/linha: {media_por_linha:.1f}s | "
                f"Restante estimado: {formatar_tempo(estimativa_restante)} | "
                f"Erros no lote: {erros}"
            )

            if batches_processados % save_every_batches == 0:
                df_parcial = (
                    pd.DataFrame(resultados_acumulados)
                    .drop_duplicates(subset=["indice_original"], keep="last")
                    .sort_values("indice_original")
                )

                salvar_csv_atomico(df_parcial, caminho_checkpoint)
                print(f"💾 Checkpoint salvo: {caminho_checkpoint}")

    df_saida = (
        pd.DataFrame(resultados_acumulados)
        .drop_duplicates(subset=["indice_original"], keep="last")
        .sort_values("indice_original")
    )

    salvar_csv_atomico(df_saida, caminho_saida)

    # Remove checkpoint apenas na extração final completa.
    if limit is None and os.path.exists(caminho_checkpoint):
        os.remove(caminho_checkpoint)

    tempo_final = time.time() - inicio_arquivo

    print("\n" + "-" * 70)
    print(f"🎉 Concluído: {nome_saida}")
    print(f"Linhas salvas: {len(df_saida)}")
    print(f"Tempo desta execução: {formatar_tempo(tempo_final)}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Extrai Coh-Metrix usando pergunta + resposta com checkpoint."
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Número de processos paralelos. Recomendo 2 no Windows."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Quantidade de linhas por lote."
    )

    parser.add_argument(
        "--save-every-batches",
        type=int,
        default=1,
        help="Salva checkpoint a cada N lotes."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processa apenas as primeiras N linhas de cada arquivo para teste."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regera o arquivo final mesmo se ele já existir."
    )

    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Apaga checkpoint anterior antes de começar."
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Processa apenas um arquivo específico. Ex.: df_11_train.csv"
    )

    args = parser.parse_args()

    print("Iniciando extração Coh-Metrix com entrada Pergunta + Resposta...")
    print(f"Pasta de dados utilizada: {PASTA_DATASET}")
    print(f"SRC utilizado: {SRC_DIR}")

    try:
        CohMetrixExtractor = importar_extrator()
        _ = CohMetrixExtractor()
        print("CohMetrixExtractor carregado com sucesso.")
    except Exception as e:
        raise ImportError(
            "Não foi possível importar/carregar o CohMetrixExtractor. "
            "Confira se o script está dentro de tools e se ../src existe."
        ) from e

    if args.file is not None:
        arquivos = [args.file]
    else:
        arquivos = ARQUIVOS_PARA_PROCESSAR

    for nome_arquivo in arquivos:
        processar_arquivo(
            nome_entrada=nome_arquivo,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            save_every_batches=args.save_every_batches,
            limit=args.limit,
            overwrite=args.overwrite,
            reset_checkpoint=args.reset_checkpoint
        )

    print("\n🏁 EXTRAÇÃO TOTAL FINALIZADA!")


if __name__ == "__main__":
    main()
