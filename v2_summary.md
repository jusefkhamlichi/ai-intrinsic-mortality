# Intrinsic Mortality PoC v2

## Strategie considerate

### 1. Senescenza cumulativa dei neuroni nascosti

Ogni neurone del layer nascosto possiede una vitalita interna che si riduce in modo irreversibile nel tempo. Il danno cresce a ogni ciclo in base alla pressione globale di invecchiamento e allo stress d'uso misurato dalle attivazioni medie. La vitalita residua controlla guadagno, precisione e sopravvivenza del neurone. Questa strategia e piu intrinseca della v1 perche introduce uno stato interno persistente, non una semplice corruzione esterna dei pesi a ogni step. Pro: leggibile, cumulativa, plausibile. Contro: richiede una piccola dinamica di stato per neurone. Questa e la strategia scelta.

### 2. Pruning irreversibile di connessioni guidato dall'uso

Le connessioni accumulano danno e vengono progressivamente spente in modo permanente. La rete perde quindi gradi di liberta strutturali, non solo precisione numerica temporanea. E piu intrinseca della v1 perche il danno e topologico e persistente. Pro: molto semplice da visualizzare e chiaramente irreversibile. Contro: puo sembrare un pruning progettato dall'esterno piu che un invecchiamento funzionale interno.

### 3. Collasso della precisione delle rappresentazioni interne

Le attivazioni latenti perdono progressivamente precisione utile: diventano piu quantizzate, piu saturate e meno affidabili. Il danno si accumula come deterioramento della qualita della rappresentazione piu che dei parametri. E piu intrinseca della v1 perche colpisce direttamente lo spazio latente interno. Pro: concettualmente forte. Contro: leggermente meno intuitiva da spiegare in un PoC molto piccolo.

## Strategia scelta

La v2 usa la strategia 1, con una piccola componente della 3:

- vitalita persistente per neurone;
- danno cumulativo guidato da eta e stress di attivazione;
- riduzione del gain;
- riduzione della precisione delle attivazioni;
- morte irreversibile dei neuroni molto deteriorati.

## Confronto sintetico v1 vs v2

### v1

- degrada i parametri effettivi usati in inferenza;
- aggiunge rumore e attenuazione dei pesi;
- spegne progressivamente neuroni nascosti;
- e convincente, ma il deterioramento puo ancora essere letto come perturbazione esterna applicata al modello.

### v2

- introduce uno stato interno persistente di senescenza;
- il danno si accumula in modo irreversibile tra uno step e il successivo;
- la qualita delle rappresentazioni interne peggiora per perdita di gain, precisione e capacita;
- il collasso finale appare piu strutturale e meno aggirabile concettualmente.

## Risultati del run verificato

- baseline iniziale: `0.956`
- v1 fase stabile: `0.954`
- v1 fase terminale: `0.373`
- v1 finale: `0.333`
- v2 fase stabile: `0.956`
- v2 fase terminale: `0.350`
- v2 finale: `0.333`

Interpretazione:

- la v2 mantiene la lunga fase stabile richiesta;
- il crollo finale resta rapido;
- la vitalita media dei neuroni scende da `1.000` a `0.022`;
- la quota di neuroni funzionalmente morti arriva a `0.938`.

## Output generati

- script: [`intrinsic_mortality_poc_v2.py`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_poc_v2.py)
- grafico: [`intrinsic_mortality_v2_results.png`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_v2_results.png)

## Commento finale

La v2 sembra davvero piu endogena della v1 perche il decadimento non e piu principalmente una manipolazione istantanea dei pesi. Il modello porta dentro di se uno stato di deterioramento che si accumula, riduce la qualita delle sue rappresentazioni interne e distrugge progressivamente la propria capacita funzionale.
