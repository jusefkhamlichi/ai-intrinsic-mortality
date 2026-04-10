# Intrinsic Mortality PoC: Sintesi v2

## Obiettivo del documento

Questo documento riassume in modo compatto:

- le 3 strategie tecniche proposte per la seconda iterazione;
- la strategia effettivamente scelta;
- i risultati comparativi tra PoC v1 e PoC v2;
- un estratto rappresentativo del codice e del README.

## Le 3 strategie proposte

### Strategia 1. Senescenza cumulativa dei neuroni nascosti

Ogni neurone del layer nascosto mantiene una variabile interna di vitalita `v_i`, compresa tra 0 e 1, che si deteriora in modo irreversibile nel tempo. Il danno cresce a ogni age step in funzione di due componenti: la pressione globale di invecchiamento e lo stress d'uso del neurone, misurato tramite la sua attivazione media. La vitalita residua modula il guadagno del neurone, la precisione della sua attivazione e la sua sopravvivenza funzionale. Questa strategia e piu intrinseca della v1 perche il danno non viene semplicemente applicato dall'esterno ai pesi: il modello trasporta uno stato interno persistente di senescenza. Pro: leggibile, cumulativa, plausibile. Contro: richiede una piccola dinamica di stato.

### Strategia 2. Pruning irreversibile di connessioni guidato dall'uso

Le connessioni sinaptiche accumulano danno e vengono spente progressivamente in modo permanente. La rete perde quindi struttura utile in modo irreversibile, riducendo gradualmente la propria capacita di rappresentazione. Questa strategia e piu intrinseca della v1 perche il decadimento diventa topologico e persistente, non solo numerico e istantaneo. Pro: semplice da visualizzare e molto netto dal punto di vista strutturale. Contro: puo sembrare un pruning progettato artificialmente piu che una vera senescenza interna.

### Strategia 3. Collasso progressivo della precisione delle rappresentazioni interne

Il danno agisce sulle attivazioni latenti piu che sui parametri, degradando la precisione utile delle rappresentazioni interne. I neuroni diventano progressivamente piu saturati, piu quantizzati e meno affidabili nel trasporto di informazione. Questa strategia e piu intrinseca della v1 perche colpisce direttamente lo spazio latente interno del modello. Pro: concettualmente molto forte. Contro: leggermente meno immediata da spiegare in un PoC minimale.

## Strategia scelta

La strategia scelta e stata la **Strategia 1**, con una piccola componente della Strategia 3.

In pratica la v2 introduce:

- una vitalita persistente per ciascun neurone nascosto;
- danno cumulativo guidato da eta e stress di attivazione;
- riduzione progressiva del gain delle attivazioni;
- riduzione della precisione delle rappresentazioni interne;
- morte irreversibile dei neuroni molto deteriorati.

### Perche questa scelta

Questa soluzione offre il miglior equilibrio tra:

- rigore concettuale;
- semplicita implementativa;
- leggibilita del codice;
- plausibilita scientifica del deterioramento.

Rispetto alla v1, il decadimento appare piu endogeno perche il modello non viene solo perturbato dall'esterno: porta dentro di se uno stato di usura che si accumula e riduce direttamente la propria capacita funzionale.

## Risultati comparativi v1 vs v2

Run verificato localmente sul dataset Iris:

| Metrica | PoC v1 | PoC v2 |
|---|---:|---:|
| Baseline iniziale | 0.956 | 0.956 |
| Accuracy media fase stabile | 0.954 | 0.956 |
| Accuracy media fase terminale | 0.373 | 0.350 |
| Accuracy finale | 0.333 | 0.333 |

### Lettura dei risultati

- entrambe le versioni mantengono una lunga fase stabile;
- entrambe mostrano un collasso rapido vicino alla fine della vita utile;
- la v2 resta molto stabile nella fase iniziale e centrale;
- la v2 collassa tramite deterioramento interno cumulativo, non principalmente tramite corruzione istantanea dei pesi.

### Indicatori interni specifici della v2

- vitalita media neuroni: `1.000 -> 0.022`
- frazione di neuroni funzionalmente morti: `0.000 -> 0.938`

Questi due indicatori rendono esplicito che nella v2 il collasso finale coincide con una quasi totale perdita della capacita interna del layer nascosto.

## Confronto concettuale v1 vs v2

### v1

- corruzione dei parametri effettivi;
- attenuazione dei pesi;
- rumore crescente;
- perdita progressiva di neuroni.

La v1 funziona bene come primo PoC, ma puo ancora essere interpretata come una degradazione applicata "da fuori" al modello.

### v2

- stato interno persistente di vitalita per neurone;
- accumulo irreversibile di danno;
- peggioramento della qualita delle attivazioni;
- perdita strutturale della capacita rappresentativa.

La v2 e piu endogena perche il danno non e solo una trasformazione dei pesi: e una dinamica interna del modello che si trascina nel tempo.

## Estratto del codice

Di seguito un estratto rappresentativo della logica centrale della v2:

```python
step_damage = pressure * (0.15 + 0.85 * stress_norm) * (0.55 + 0.45 * (1.0 - vitality))
cumulative_damage += step_damage
vitality = np.exp(-1.6 * cumulative_damage)
vitality = np.clip(vitality, 0.0, 1.0)

gain = vitality ** 1.3
ceiling = 2.5 * (0.2 + vitality)
precision_levels = np.maximum(2, np.round(3 + 13 * vitality)).astype(np.int64)
```

Significato:

- il danno cresce a ogni ciclo di vita;
- la vitalita non recupera;
- neuroni meno vitali amplificano meno il segnale;
- le attivazioni diventano piu compresse e meno precise;
- il decadimento e quindi interno, cumulativo e funzionale.

## Estratto dal README / descrizione del meccanismo

Estratto concettuale sintetico:

```text
W_eff = health(t)^2 * W + sigma(t) * N
b_eff = health(t)^2 * b + sigma(t) * n
sigma(t) = noise_max * (1 - health(t)) ^ noise_power
```

Questo era il cuore della v1: utile per mostrare il collasso, ma ancora relativamente vicino a una corruzione esterna dei parametri.

La v2 sposta invece il centro del decadimento verso:

- vitalita persistente dei neuroni;
- danno cumulativo;
- perdita di precisione interna;
- morte funzionale irreversibile.

## File di riferimento

- [`intrinsic_mortality_poc.py`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_poc.py)
- [`intrinsic_mortality_poc_v2.py`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_poc_v2.py)
- [`README.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/README.md)
- [`v2_summary.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/v2_summary.md)

## Conclusione

La seconda iterazione mantiene il comportamento desiderato:

- lunga fase stabile;
- collasso rapido finale.

Ma lo fa con un meccanismo piu endogeno della v1, perche il deterioramento non e solo applicato ai parametri: emerge da una senescenza interna persistente che distrugge progressivamente la capacita rappresentativa del modello.
