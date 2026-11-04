"""
Funções de extração usadas pelos processos paralelos.

Ficam em um módulo separado (e não no script principal) por um motivo prático:
quando a função está no próprio script executado, o joblib/loky a reenvia a cada
tarefa e o "cache" do extrator é perdido, recriando o CohMetrixExtractor em quase
toda linha. Aqui o extrator é criado UMA vez por processo.

A lógica de extração e as colunas geradas são as mesmas de
extracao_asag_tcc_cohmetrix_qa_progress.py.
"""
import os
import sys

import pandas as pd

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(TOOLS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

_EXTRATOR = None


def obter_extrator():
    global _EXTRATOR
    if _EXTRATOR is None:
        from aibox.nlp.features.portuguese.cohmetrix import CohMetrixExtractor
        _EXTRATOR = CohMetrixExtractor()
    return _EXTRATOR


def construir_texto_qa(question_text, answer_text):
    q = str(question_text) if pd.notna(question_text) else ""
    a = str(answer_text) if pd.notna(answer_text) else ""
    return q + "\n Resposta: " + a


def extrair_metricas_linha(item):
    """item: dict com indice_original, texto_combinado, nota, id?, question_id?"""
    metricas = {
        "resposta_original": item["texto_combinado"],
        "nota_original": item["nota"],
        "indice_original": item["indice_original"],
    }
    if "id" in item:
        metricas["id"] = item["id"]
    if "question_id" in item:
        metricas["question_id"] = item["question_id"]

    texto_limpo = str(item["texto_combinado"]).strip()
    if len(texto_limpo) < 2 or not any(c.isalpha() for c in texto_limpo):
        return metricas

    try:
        resultado = obter_extrator().extract(item["texto_combinado"])
        metricas.update(resultado.as_dict())
    except Exception as e:  # registra o erro e segue; a linha é tentada de novo no fim
        metricas["erro_cohmetrix"] = str(e)
    return metricas
