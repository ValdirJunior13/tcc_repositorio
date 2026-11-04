"""
Extrai o Coh-Metrix (pergunta + resposta) SOMENTE para o df_12_test, com checkpoint.

Coloque este arquivo e o cohmetrix_worker.py na pasta tools/ (a mesma do
extracao_asag_tcc_cohmetrix_qa_progress.py) e rode a partir da raiz do projeto:

    python .\\tools\\extrair_cohmetrix_df12_test.py --limit 20     # teste rápido
    python .\\tools\\extrair_cohmetrix_df12_test.py                # extração completa

Pode interromper a qualquer momento (Ctrl+C ou fechar a janela). Ao rodar o mesmo
comando de novo, ele continua de onde parou.

Arquivos em dataset/anderson:
  lê     df_12_test.csv
  cria   CHECKPOINT_cohmetrix_df_12_test_QA.csv   (progresso; salvo a cada lote)
  cria   cohmetrix_df_12_test_QA.csv              (resultado completo)
  cria   cohmetrix_df_12_test_BACKUP_so_resposta.csv (cópia do arquivo antigo, uma vez só)
  troca  cohmetrix_df_12_test.csv                 (só no fim, se a validação passar)

Nenhum outro arquivo é alterado.
"""
import argparse
import os
import re
import shutil
import sys
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=RuntimeWarning)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
PASTA = os.path.join(ROOT_DIR, "dataset", "anderson")
if not os.path.isdir(PASTA):
    PASTA = os.path.join(ROOT_DIR, "dataset")

# Os processos paralelos precisam encontrar cohmetrix_worker.py e a pasta src.
for p in (SRC_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [TOOLS_DIR, SRC_DIR] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]
)

import cohmetrix_worker as W  # noqa: E402

ENTRADA = "df_12_test.csv"
ARQ_ATUAL = "cohmetrix_df_12_test.csv"
ARQ_BACKUP = "cohmetrix_df_12_test_BACKUP_so_resposta.csv"
ARQ_REFERENCIA_COLUNAS = "cohmetrix_df_12_train.csv"


def caminho(nome):
    return os.path.join(PASTA, nome)


def limpar(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def tempo(s):
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def salvar_atomico(registros, destino):
    df = pd.DataFrame(list(registros.values())).sort_values("indice_original")
    tmp = destino + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, destino)
    return df


def tem_erro(reg):
    e = reg.get("erro_cohmetrix")
    return e is not None and not (isinstance(e, float) and pd.isna(e)) and str(e).strip() != ""


def preparar_itens(df):
    itens = []
    for idx, row in df.iterrows():
        item = {
            "indice_original": int(idx),
            "texto_combinado": W.construir_texto_qa(row["question_text"], row["answer_text"]),
            "nota": row["grade"],
        }
        if "id" in df.columns:
            item["id"] = row["id"]
        if "question_id" in df.columns:
            item["question_id"] = row["question_id"]
        itens.append(item)
    return itens


def carregar_checkpoint(arq):
    if not os.path.exists(arq):
        return {}
    try:
        df = pd.read_csv(arq)
    except Exception as e:
        print(f"AVISO: checkpoint ilegível ({e}). Renomeando para .corrompido e começando do zero.")
        os.replace(arq, arq + ".corrompido")
        return {}
    regs = {}
    for r in df.to_dict("records"):
        regs[int(r["indice_original"])] = r
    print(f"Checkpoint encontrado: {len(regs)} linhas já processadas.")
    return regs


def rodar_lotes(itens, registros, arq_ckpt, n_jobs, batch_size, rotulo, total_arquivo):
    """Processa os itens em lotes, salvando o checkpoint após cada lote."""
    if not itens:
        return
    n_lotes = (len(itens) + batch_size - 1) // batch_size
    inicio = time.time()
    feitos = 0
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for k in range(n_lotes):
            lote = itens[k * batch_size:(k + 1) * batch_size]
            t0 = time.time()
            res = parallel(delayed(W.extrair_metricas_linha)(it) for it in lote)
            for r in res:
                idx = int(r["indice_original"])
                # Numa nova tentativa, só substitui se a nova extração não falhou.
                if idx not in registros or tem_erro(registros[idx]) or not tem_erro(r):
                    registros[idx] = r
            salvar_atomico(registros, arq_ckpt)
            feitos += len(lote)
            media = (time.time() - inicio) / feitos
            erros = sum(tem_erro(r) for r in res)
            print(
                f"[{rotulo}] lote {k + 1}/{n_lotes} em {tempo(time.time() - t0)} | "
                f"concluídas {len(registros)}/{total_arquivo} | "
                f"{media:.1f}s/linha | restante ~{tempo(media * (len(itens) - feitos))} | "
                f"erros no lote: {erros} | checkpoint salvo",
                flush=True,
            )


def validar(df_entrada, df_saida):
    problemas = []
    if len(df_saida) != len(df_entrada):
        problemas.append(f"linhas: {len(df_saida)} (esperado {len(df_entrada)})")
    if set(df_saida["indice_original"].astype(int)) != set(range(len(df_entrada))):
        problemas.append("índices faltando ou sobrando")
    chaves_qa = set(
        limpar(W.construir_texto_qa(q, a))
        for q, a in zip(df_entrada["question_text"], df_entrada["answer_text"])
    )
    taxa = df_saida["resposta_original"].map(limpar).isin(chaves_qa).mean()
    if taxa < 1.0:
        problemas.append(f"só {taxa:.1%} dos textos casam com pergunta + resposta")
    n_erros = int(df_saida.apply(lambda r: tem_erro(r.to_dict()), axis=1).sum()) if "erro_cohmetrix" in df_saida else 0
    if n_erros:
        problemas.append(f"{n_erros} linhas com erro de extração")

    ref = caminho(ARQ_REFERENCIA_COLUNAS)
    if os.path.exists(ref):
        cols_ref = set(pd.read_csv(ref, nrows=1).columns) - {"erro_cohmetrix"}
        cols_novo = set(df_saida.columns) - {"erro_cohmetrix"}
        if cols_ref != cols_novo:
            problemas.append(
                f"colunas diferentes do {ARQ_REFERENCIA_COLUNAS}: "
                f"faltam {sorted(cols_ref - cols_novo)} | sobram {sorted(cols_novo - cols_ref)}"
            )
    metricas = [c for c in df_saida.columns if c not in
                {"resposta_original", "nota_original", "indice_original", "id", "question_id", "erro_cohmetrix"}]
    vazias = int(df_saida[metricas].isna().all(axis=1).sum()) if metricas else len(df_saida)
    print(f"\nValidação: {len(df_saida)} linhas | casamento pergunta+resposta {taxa:.1%} | "
          f"{len(metricas)} métricas | linhas sem nenhuma métrica: {vazias} | erros: {n_erros}")
    return problemas


def main():
    ap = argparse.ArgumentParser(description="Coh-Metrix (pergunta + resposta) do df_12_test com checkpoint.")
    ap.add_argument("--n-jobs", type=int, default=2, help="Processos paralelos (2 é seguro no Windows).")
    ap.add_argument("--batch-size", type=int, default=20, help="Linhas por lote (checkpoint a cada lote).")
    ap.add_argument("--limit", type=int, default=None, help="Teste: processa só as N primeiras linhas.")
    ap.add_argument("--tentativas-erro", type=int, default=2, help="Novas tentativas para linhas com erro.")
    ap.add_argument("--nao-instalar", action="store_true",
                    help="Não substitui cohmetrix_df_12_test.csv no final.")
    args = ap.parse_args()

    teste = args.limit is not None
    sufixo = f"TEST_limit{args.limit}_" if teste else ""
    arq_ckpt = caminho(f"CHECKPOINT_{sufixo}cohmetrix_df_12_test_QA.csv")
    arq_saida = caminho(f"{sufixo}cohmetrix_df_12_test_QA.csv")

    print(f"Pasta de dados: {PASTA}")
    print(f"Checkpoint:     {arq_ckpt}")
    print(f"Saída:          {arq_saida}")

    df = pd.read_csv(caminho(ENTRADA))
    if teste:
        df = df.head(args.limit).copy()
        print(f"MODO TESTE: {len(df)} linhas; nada será instalado.")
    itens = preparar_itens(df)
    total = len(itens)

    print("Carregando o CohMetrixExtractor (teste de importação)...")
    W.obter_extrator()
    print("OK.")

    registros = carregar_checkpoint(arq_ckpt)
    pendentes = [it for it in itens if it["indice_original"] not in registros]
    print(f"Total: {total} | já feitas: {total - len(pendentes)} | pendentes: {len(pendentes)}")

    try:
        rodar_lotes(pendentes, registros, arq_ckpt, args.n_jobs, args.batch_size, "extração", total)
        for t in range(1, args.tentativas_erro + 1):
            com_erro = [it for it in itens if tem_erro(registros[it["indice_original"]])]
            if not com_erro:
                break
            print(f"\nNova tentativa {t}/{args.tentativas_erro} para {len(com_erro)} linhas com erro...")
            rodar_lotes(com_erro, registros, arq_ckpt, args.n_jobs, args.batch_size, f"tentativa {t}", total)
    except KeyboardInterrupt:
        if registros:
            salvar_atomico(registros, arq_ckpt)
        print(f"\nInterrompido. Progresso salvo ({len(registros)}/{total}). Rode o mesmo comando para continuar.")
        sys.exit(1)

    df_saida = salvar_atomico(registros, arq_saida)
    print(f"\nArquivo completo salvo: {arq_saida}")

    problemas = validar(df, df_saida)
    if problemas:
        print("NÃO INSTALADO. Problemas encontrados:")
        for p in problemas:
            print("  -", p)
        print("O checkpoint foi mantido. Rode de novo para tentar as linhas com erro.")
        sys.exit(2)

    if teste or args.nao_instalar:
        print("Validação OK. Arquivo não instalado (modo teste ou --nao-instalar).")
        return

    atual, backup = caminho(ARQ_ATUAL), caminho(ARQ_BACKUP)
    if os.path.exists(atual) and not os.path.exists(backup):
        shutil.copy2(atual, backup)
        print(f"Backup do arquivo antigo: {backup}")
    shutil.copy2(arq_saida, atual)
    os.remove(arq_ckpt)
    print(f"Instalado: {atual} agora usa pergunta + resposta. Checkpoint removido.")


if __name__ == "__main__":
    main()
