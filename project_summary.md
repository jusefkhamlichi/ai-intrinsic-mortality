# Intrinsic Mortality PoC: Documento Riassuntivo

## Obiettivo

Questo proof of concept dimostra una forma minimale di "mortalita intrinseca" in una rete neurale semplice:

- il modello viene addestrato normalmente;
- resta utile per la maggior parte della sua vita;
- entra poi in una fase di degrado rapido;
- termina in una condizione quasi inutilizzabile.

L'obiettivo non era costruire un sistema industriale, ma una dimostrazione chiara, leggibile e scientificamente plausibile del meccanismo di decadimento.

## Strategia tecnica

La strategia scelta e stata guidata da quattro criteri:

- mantenere il codice molto piccolo;
- usare un dataset semplice e built-in;
- implementare il decadimento dentro il modello, non come semplice blocco esterno;
- ottenere una curva di salute quasi piatta per lungo tempo e con crollo rapido finale.

Per questo il PoC usa:

- dataset `Iris`;
- MLP minima;
- variabile di eta esplicita;
- funzione `health(t)` sigmoide con collasso tardivo;
- degrado interno tramite pesi corrotti e perdita progressiva di neuroni nascosti.

## Dataset scelto

Il dataset utilizzato e `Iris` da `scikit-learn`.

Ragioni della scelta:

- non richiede download;
- e molto noto e facile da spiegare;
- permette di addestrare rapidamente un modello piccolo;
- e sufficiente per osservare in modo pulito la perdita di performance.

Caratteristiche del setup:

- 150 osservazioni totali;
- 4 feature numeriche;
- 3 classi;
- split del PoC: 105 train, 45 test.

## Architettura del modello

Il modello e una rete feed-forward molto semplice:

```text
Input (4) -> Linear -> ReLU -> Linear -> Output (3)
```

Dettagli:

- input: 4 dimensioni;
- hidden layer: 16 neuroni;
- attivazione: ReLU;
- output: 3 logit, uno per classe.

Questa architettura e stata scelta perche:

- e leggibile;
- e abbastanza potente da classificare bene Iris;
- rende chiaro il rapporto tra danno interno e crollo della performance.

## Scelte implementative

La preferenza era PyTorch, ma nell'ambiente locale `torch` non risultava installato. Per mantenere il PoC immediatamente eseguibile end-to-end senza setup aggiuntivo, l'implementazione e stata fatta in NumPy puro.

Questa scelta conserva i requisiti essenziali del progetto:

- un singolo file eseguibile;
- pieno controllo su training e forward;
- massima leggibilita del meccanismo di decadimento.

## Addestramento

Il modello viene addestrato con una pipeline standard:

- standardizzazione delle feature;
- training su train set;
- loss cross-entropy;
- ottimizzazione SGD semplice;
- lieve regolarizzazione L2.

L'obiettivo del training e ottenere un modello inizialmente sano. Nel run validato localmente, la baseline di test ha raggiunto `0.956`.

## Variabile di eta e health function

Dopo il training, il modello entra in una simulazione di vita utile con `age` che varia da `0` a `T`.

La salute del modello e descritta da:

```text
health(t) = 1 / (1 + exp(k * (t / T - c)))
```

dove:

- `T` e la vita massima simulata;
- `c` e il punto di onset del collasso;
- `k` controlla la ripidita della discesa.

Questa funzione e stata scelta per ottenere esattamente il comportamento richiesto:

- salute alta e quasi piatta per gran parte della vita;
- calo molto rapido vicino alla fine;
- valore finale basso nella fase terminale.

## Meccanismo di decadimento intrinseco

Il decadimento viene applicato dentro il modello, modificando i parametri effettivi usati in inferenza.

Formula principale:

```text
W_eff = health(t)^2 * W + sigma(t) * N
b_eff = health(t)^2 * b + sigma(t) * n
sigma(t) = noise_max * (1 - health(t)) ^ noise_power
```

Interpretazione:

- all'inizio `health(t)` e circa 1, quindi il modello usa quasi i pesi originali;
- quando la salute cala, il contributo dei pesi addestrati si attenua;
- il rumore cresce in modo non lineare;
- nella fase finale il rumore domina e la rete perde affidabilita.

Questo soddisfa il requisito di un decadimento endogeno applicato dentro il modello e non come semplice post-processing esterno.

## Perdita progressiva di neuroni nascosti

Per rendere il collasso terminale piu netto, il PoC aggiunge un secondo meccanismo semplice:

- il layer nascosto perde progressivamente neuroni attivi;
- i neuroni vengono "persi" secondo un ordine di vulnerabilita fisso;
- il numero di neuroni sopravvissuti e proporzionale alla health.

Questo introduce una vera perdita di capacita interna:

- all'inizio quasi tutti i neuroni sono disponibili;
- nella fase finale restano pochi neuroni e anche i pesi sono fortemente corrotti;
- la combinazione di rumore e riduzione di capacita accelera il collasso.

## Flusso end-to-end dello script

Lo script esegue questi passaggi:

1. carica il dataset Iris;
2. esegue train/test split;
3. standardizza i dati;
4. addestra la rete;
5. salva i parametri sani di riferimento;
6. simula l'invecchiamento per tutti gli age step;
7. costruisce per ogni eta una versione degradata del modello;
8. misura accuracy, health e damage ratio;
9. genera un grafico riepilogativo;
10. stampa un summary finale in console.

## Output prodotti

Lo script produce:

- un riepilogo testuale in console;
- un grafico salvato su file.

Il grafico `intrinsic_mortality_results.png` contiene:

- accuracy di test vs eta;
- `health(t)` vs eta;
- danno normalizzato vs eta.

Il riepilogo in console riporta:

- dataset e architettura;
- formula del decadimento;
- performance iniziale;
- performance media nella fase stabile;
- performance media nella fase terminale;
- performance finale;
- health iniziale e finale;
- damage ratio iniziale e finale;
- eta di onset del collasso.

## Risultati osservati

Nel run eseguito localmente:

- performance iniziale: `0.956`
- fase stabile: media `0.954`
- fase terminale: media `0.373`
- performance finale: `0.333`

Interpretazione:

- il modello resta utile e quasi invariato per gran parte della vita;
- il degrado resta marginale nella fase stabile;
- il crollo finale e rapido e chiaramente visibile;
- la performance finale scende al livello casuale atteso per 3 classi.

## Perche il PoC funziona

Il PoC e convincente perche il calo di performance non e artificiale o cosmetico. Deriva da un meccanismo esplicito e misurabile:

- salute interna che evolve nel tempo;
- pesi effettivi sempre piu corrotti;
- perdita progressiva di capacita del layer nascosto;
- relazione chiara tra salute, danno e accuracy.

In sintesi, il modello non viene semplicemente "spento": invecchia e collassa.

## Limiti attuali

Questo PoC e intenzionalmente minimale. I principali limiti sono:

- dataset piccolo;
- implementazione NumPy invece di PyTorch;
- nessuna analisi multi-seed;
- nessuna valutazione statistica avanzata;
- meccanismo di decadimento astratto, non biologico o hardware-realistic.

Questi limiti sono accettabili in questa fase perche l'obiettivo era la dimostrazione concettuale.

## Possibili estensioni

Passi successivi naturali:

- porting in PyTorch;
- uso di MNIST o dataset sintetici piu complessi;
- confronto tra diverse health function;
- separazione di piu tipi di danno;
- piu run con seed diversi;
- trasformazione del PoC in repo pubblico con struttura piu formale.

## File del progetto

- [`intrinsic_mortality_poc.py`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_poc.py)
- [`README.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/README.md)
- [`project_summary.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/project_summary.md)
- [`intrinsic_mortality_results.png`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_results.png)

## Conclusione

Il progetto dimostra che una rete neurale semplice puo essere progettata per avere una vita utile lunga e stabile, seguita da un decadimento accelerato e da una morte funzionale finale.

La soluzione e compatta, leggibile e gia adatta come base per una seconda iterazione piu rigorosa o per la pubblicazione iniziale del concept.
