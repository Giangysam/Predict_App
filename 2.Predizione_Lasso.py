 # Usa SOLO il modello Logistic Regression ("Lasso" nello script di confronto) per
# prevedere un nuovo record e spiegarlo con SHAP. Da eseguire ogni volta che serve
# una previsione; lo script di confronto (RF/XGB/Lasso) resta quello da rilanciare
# una volta al mese per validare che questo sia ancora il modello giusto da usare.
#
# ATTENZIONE ereditata dallo script di confronto: la pipeline sotto usa
# solver="liblinear" con l1_ratio=1.0 ma SENZA penalty="l1" esplicito. In
# scikit-learn il default di penalty e' "l2", e l1_ratio ha effetto solo con
# penalty="elasticnet": quindi il modello chiamato "Lasso" e' in realta' una
# logistic regression L2 (tipo Ridge), non L1. Le previsioni non ne risentono,
# ma i coefficienti non si azzerano automaticamente (nessuna vera selezione
# delle feature). Se serve L1 vera, aggiungere penalty="l1" qui sotto — ma
# questo la rende un modello diverso da quello validato nel confronto mensile,
# quindi andrebbe rivalidato.

#import json
import sys
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==== stessa configurazione dello script di confronto ====
CAT_COLS = ["W", "V", "Z", "AA"]   # Colonne categoriche ad alta cardinalità (ID basi/postazioni) — OneHotEncoder.
COL_ORDINALE = "R"  # trattata come numerica: l'ordine ha significato, un one-hot lo distruggerebbe
COLS_DA_ELIMINARE = ["SOGGETTO", "STAT1"]    # Colonne da eliminare
CSV_PATH = "https://raw.githubusercontent.com/Giangysam/Correlations/main/DATI_RIS_coperti.xlsx"
CSV_NUOVO_RECORD = "INSERIRE IL PATH DEL FILE DA ESAMINARE.csv"  # 1 riga, stesse colonne/sep/decimal di DATI.csv
TARGET_COL = "TARGET"
SEP = ";"
DECIMAL = ","
RANDOM_STATE = 42


def carica_dati(path=CSV_PATH):
    df = pd.read_csv(path, sep=SEP, decimal=DECIMAL)
    df = df.drop(columns=COLS_DA_ELIMINARE, errors="ignore")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL]).copy()
    
    # Categoriche -> stringa
    for col in [c for c in CAT_COLS if c in X.columns]:
        X[col] = X[col].astype(str).str.strip()
    
    # Tutto il resto -> numerico    
    for col in [c for c in X.columns if c not in CAT_COLS]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def individua_colonne(X, cat_cols=CAT_COLS):
    categoriche = [c for c in cat_cols if c in X.columns]
    numeriche = [c for c in X.columns if c not in categoriche]
    
    binarie, continue_ = [], []
    for c in numeriche:
        valori = X[c].dropna().unique()
        if len(valori) <= 2 and set(np.unique(valori)).issubset({0, 1}):
            binarie.append(c)
        else:
            continue_.append(c)
    return categoriche, continue_, binarie


def costruisci_pipeline_lasso(categoriche, colonne_continue, colonne_binarie):
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())]), colonne_continue),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoriche),
        ("bin", SimpleImputer(strategy="most_frequent"), colonne_binarie),
    ], remainder="drop")
    clf = LogisticRegression(
        solver="liblinear", l1_ratio=1.0, C=0.1,  # vedi ATTENZIONE 
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    return Pipeline([("prep", preprocessor), ("clf", clf)])


#   QUESTE RIGHE SERVONO A CARICARE UN FILE JSON DA APP CREATE ESTERNAMENTE
#def carica_nuovo_record_da json(json_string: str, colonne_attese: list, categoriche: list) -> pd.DataFrame:
#    record_dict = json.loads(json_string) # passa la stringa Json in un dizionario Python
#    return prepara_nuovo_record(record_dict, colonne_attese, categoriche)


def carica_nuovo_record_da_csv(path: str = CSV_NUOVO_RECORD, sep: str = SEP, decimal: str = DECIMAL,
                                riga: int = 0) -> dict:
    #Legge il CSV del nuovo oggetto (stesso formato di DATI_RIS.csv) e restituisce una riga come dict.
    #dtype=str sulle colonne categoriche: evita che pandas le interpreti come numeri e ne perda la
    #formattazione (es. zeri iniziali in BASE_ID/HB_ID) rispetto a come erano in fase di training
    
    df = pd.read_csv(path, sep=sep, decimal=decimal, dtype={c: str for c in CAT_COLS})
    if df.empty:
        raise ValueError(f"'{path}' non contiene righe.")
    if len(df) > 1:
        print(f"ATTENZIONE: '{path}' contiene {len(df)} righe, valuto solo la riga {riga} (indice da 0).")
    return df.iloc[riga].to_dict()


def prepara_nuovo_record(record: dict, colonne_attese: list, categoriche: list) -> pd.DataFrame:
    df = pd.DataFrame([record])
    mancanti = [c for c in colonne_attese if c not in df.columns]
    if mancanti:
        raise ValueError(f"Colonne mancanti nel nuovo record: {mancanti}")
    df = df[colonne_attese].copy()
    for col in categoriche:
        df[col] = df[col].astype(str).str.strip()
    for col in [c for c in colonne_attese if c not in categoriche]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def valuta_record(nuovo_record: dict, top_n: int = 15):
    X, y = carica_dati()
    categoriche, colonne_continue, colonne_binarie = individua_colonne(X)

    pipe = costruisci_pipeline_lasso(categoriche, colonne_continue, colonne_binarie)
    pipe.fit(X, y)  # fit finale su TUTTI i dati disponibili, non solo il train set del confronto mensile

    df_nuovo = prepara_nuovo_record(nuovo_record, list(X.columns), categoriche)

    proba = float(pipe.predict_proba(df_nuovo)[0, 1])
    # Soglia di default (0.5): e' quella con cui lo script di confronto ha calcolato
    # il recall/precision su cui si e' scelto il modello. Cambiarla qui senza
    # ricalcolare quelle metriche a una nuova soglia renderebbe il confronto invalido.
    classe = int(pipe.predict(df_nuovo)[0])

    prep = pipe.named_steps["prep"]
    modello = pipe.named_steps["clf"]
    feature_names = prep.get_feature_names_out()

    # LinearExplainer: per un modello lineare da' i contributi SHAP esatti (nessuna
    # approssimazione Kernel/Sampling necessaria per RF/XGB).
    sfondo = prep.transform(X)
    x_nuovo_t = prep.transform(df_nuovo)
    explainer = shap.LinearExplainer(modello, sfondo)
    shap_values = np.asarray(explainer.shap_values(x_nuovo_t))[0]

    contributi = pd.Series(shap_values, index=feature_names)
    contributi = contributi.reindex(contributi.abs().sort_values(ascending=False).index)
    
    # QUESTE RIGHE SI INSERISCONO SE SI SCEGLIE DI REALIZZARE UN APP, PER RESTITUIRE I DATI IN JSON
    # results = {
    #    'probabilità difetto': proba,
    #    'esito_classe': classe,
    #    'contributi_shap': contributi.head(top_n).to_dict()         # CONVERTE ILA SERIES IN DIZIONARIO PER JSON
    #}
    #return results      # LA FUNZIONE RESTITUISCE IL DIZIONARIO DEI RISULTATI


    print(f"Probabilita' di difetto: {proba:.1%}")
    print("Esito (soglia 0.5):", "POSSIBILE DIFETTO" if classe == 1 else "nessun difetto rilevato")
    print(f"\nContributi SHAP (top {top_n}, positivo = spinge verso il difetto):")
    print(contributi.head(top_n).to_string())

    return proba, classe, contributi


if __name__ == "__main__":
    # Percorso del CSV col nuovo record: da riga di comando (py predici_con_lasso.py percorso.csv)
    # oppure, se omesso, CSV_NUOVO_RECORD sopra.
    percorso = sys.argv[1] if len(sys.argv) > 1 else CSV_NUOVO_RECORD
    nuovo_record = carica_nuovo_record_da_csv(percorso)
    valuta_record(nuovo_record)


# QUESTE RIGHE SI INSERISCONO PER USARE L'APPLICAZIONE CON VISUAL STUDIO
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        input_json_string = sys.argv[1]
#        
#        X_template, _ = carica_dati()
#        colonne_attese = list (X_template.columns)
#        categoriche_in_X = [c for c in CAT_COLS if c in X_template.columns]
#        nuovo_record_df = carica_nuovo_record_da_json(input_json_string, colonne_attese, categoriche_in_X)
#        
#        final_results = valuta_record(nuovo_record_df, colonne_attese, categoriche_in_X)
#        
#        print(json.dumps(final_results))
#    else:
#        print(json.dumps({'error': 'Nessun record JSON fornito come argomento.'}), file=sys.stderr)
#        sys.exit(1)
