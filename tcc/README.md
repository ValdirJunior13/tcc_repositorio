# TCC — Avaliação Automática de Respostas Curtas em Português Brasileiro

Código e resultados do Trabalho de Conclusão de Curso **"Avaliação Automática de Respostas Curtas em Português Brasileiro: uma análise comparativa de representações textuais e modelos de regressão"** (Engenharia da Computação — UFRPE/UABJ, 2026).

Autor: Valdir Zacarias de Oliveira Junior · Orientador: Anderson Pinheiro Cavalcanti

## O que tem aqui

```text
tcc/
├── README.md
├── requirements.txt
├── notebooks/
│   └── resultados_tcc.ipynb      # reconstrói as Tabelas 3 a 10 a partir dos CSVs (não treina nada)
├── resultados/
│   ├── df_11/                    # experimento principal e experimento restrito — df_11
│   ├── df_12/                    # experimento principal — df_12 (após a correção do Coh-Metrix)
│   └── verificacao/              # saída da verificação da coluna indice_original (df_11)
└── scripts/                      # código usado para gerar os resultados
```

**Para ver os resultados**, abra `notebooks/resultados_tcc.ipynb`. Ele já está executado e mostra as tabelas do Capítulo 4. Para executar de novo, basta ter `pandas` instalado.

## Scripts

| Etapa | Script | Gera |
|---|---|---|
| Extração do Coh-Metrix (pergunta + resposta) para df_11 e df_12 (treino) | `extracao_asag_tcc_cohmetrix_qa_progress.py` | `cohmetrix_df_*.csv` |
| Extração do Coh-Metrix do df_12 (teste), com checkpoint | `extrair_cohmetrix_df12_test.py` (+ `cohmetrix_worker.py`) | `cohmetrix_df_12_test.csv` |
| Experimento principal — df_11 (MiniLM) | `analise_asag_mvp_validacao_cohmetrix_qa.py` | `resultados/df_11/resultados_df_11.csv` |
| Experimento principal — df_11 (BERTimbau) | `analise_bert_novo_modelo_validacao_cohmetrix_qa.py` | `resultados/df_11/resultados_df_11_bertimbau.csv` |
| Experimento restrito — df_11 (MiniLM / BERTimbau) | `analise_semelhante_mvp_validacao_cohmetrix_qa.py`, `analise_semelhante_bert_validacao_cohmetrix_qa.py` | `resultados/df_11/restrito_*.csv` |
| Experimento principal — df_12 (MiniLM / BERTimbau) | `analise_asag_mvp_validacao_cohmetrix_qa_corrigido.py`, `analise_bert_novo_modelo_validacao_cohmetrix_qa_corrigido.py` | `resultados/df_12/*.csv` |
| Verificação da coluna `indice_original` no df_11 | `verificar_indice_original_df11.py` | `resultados/verificacao/*.txt` |

As versões `_corrigido` diferem das demais apenas em quatro pontos, marcados com `[CORRIGIDO]` no código: não usam as colunas auxiliares `indice_original` e `erro_cohmetrix` como características; interrompem a execução se algum registro não for associado ao Coh-Metrix (em vez de preencher com zero); exigem as mesmas métricas, na mesma ordem, no treino e no teste; e gravam a saída em uma pasta separada. Modelos, hiperparâmetros, TF-IDF, *embeddings* e validação interna são os mesmos.

## Sobre o Coh-Metrix

As características do Coh-Metrix foram extraídas da mesma entrada usada pelo TF-IDF e pelos *embeddings*: `enunciado da questão + "\n Resposta: " + resposta do estudante`. Dois pontos estão documentados na Seção 3.4.3 do TCC:

- **df_11:** o vetor de características incluiu a coluna auxiliar `indice_original` (posição do registro no arquivo). Como os arquivos estão ordenados por questão, ela funcionou como indicador indireto da questão. Sem ela, o R² da configuração TF-IDF + Coh-Metrix com HistGradientBoosting cai de 0,2619 para 0,0314 (ver `resultados/verificacao/`). Por isso, os resultados do df_11 com Coh-Metrix aparecem marcados com `*` e não são usados para escolher o melhor resultado.
- **df_12:** o arquivo de teste do Coh-Metrix foi reextraído com a entrada pergunta + resposta, e os experimentos do df_12 foram executados sem a coluna auxiliar.

## Como reproduzir

1. Crie um ambiente Python 3.11 e instale as dependências:

   ```bash
   pip install -r tcc/requirements.txt
   python -m spacy download pt_core_news_lg
   ```

   A extração do Coh-Metrix usa o `CohMetrixExtractor` da biblioteca AiBox NLP, incluída neste repositório em `nlp-main/`. Instale-a a partir da raiz do repositório:

   ```bash
   pip install -e "./nlp-main[br]"
   ```

2. Coloque os dados em `tcc/dataset/anderson/` (a pasta não é versionada):

   ```text
   tcc/dataset/anderson/
     df_11_train.csv   df_11_test.csv   df_12_train.csv   df_12_test.csv
     cohmetrix_df_11_train.csv   cohmetrix_df_11_test.csv
     cohmetrix_df_12_train.csv   cohmetrix_df_12_test.csv
   ```

   Os arquivos de treino e teste têm as colunas `id`, `question_id`, `question_text`, `answer_text` e `grade`. Os arquivos `cohmetrix_*` podem ser gerados pelos scripts de extração.

3. Execute os scripts a partir da raiz do repositório, por exemplo:

   ```bash
   python tcc/scripts/analise_bert_novo_modelo_validacao_cohmetrix_qa_corrigido.py df_12
   python tcc/scripts/verificar_indice_original_df11.py tcc/dataset/anderson
   ```

   Os scripts `_corrigido` aceitam o nome do particionamento como argumento (`df_11`, `df_12` ou ambos); os demais executam os dois particionamentos. Diferenças pequenas nos valores podem ocorrer com versões diferentes do spaCy e das demais bibliotecas.

## Dados

Os experimentos usam a versão 2 do conjunto **pt-asag-2018** (9.862 respostas de Biologia, notas de 0 a 3), descrito por Galhardi, Souza e Brancher (2020). Os particionamentos `df_11` (questões inéditas no teste) e `df_12` (mesmas questões no treino e no teste) foram preparados durante a orientação e **não estão incluídos neste repositório**.

## Créditos

- Biblioteca AiBox NLP (licença MIT), usada para a extração das características do Coh-Metrix.
- Coh-Metrix PT-BR: Camelo, Justino e Mello (2020).
- Conjunto de dados pt-asag-2018: Galhardi, Souza e Brancher (2020).

## Histórico

A pasta `nlp-main/tools/` mantém as versões exploratórias e intermediárias dos scripts, desenvolvidas ao longo da pesquisa. Os scripts finais, usados nos resultados do TCC, são os da pasta `tcc/scripts/`.
