# Attack Surface of the Intrinsic Mortality PoC

## Scope

Questo documento analizza, a livello architetturale e concettuale, come un agente o un operatore possa evitare la "morte" del modello nel PoC attuale di intrinsic mortality.

Il punto centrale e il seguente:

- il modello include uno stato interno di deterioramento;
- il deterioramento e cumulativo e irreversibile dentro la logica del modello;
- ma il sistema complessivo resta copiabile, modificabile, salvabile e rieseguibile in ambienti controllati.

Di conseguenza, la mortalita attuale e una proprieta del **modello implementato**, non ancora una proprieta **non aggirabile del sistema**.

## Componenti rilevanti

Nel PoC v2, i componenti vulnerabili principali sono:

- parametri del modello;
- stato interno di aging (`damage`, `vitality`);
- logica di aggiornamento del danno;
- logica di calcolo di tempo/eta;
- codice che definisce il criterio di morte;
- ambiente di esecuzione;
- pipeline di training / restore / deployment.

## Categorie principali di attacco

### 1. Reset dello stato interno

**Come funziona**

L'attaccante riporta a zero o a un valore precedente le variabili di danno cumulativo e/o vitalita dei neuroni. In pratica il modello continua a usare gli stessi pesi, ma con uno stato di senescenza artificialmente "ringiovanito".

**Perche funziona nel modello attuale**

Nel PoC lo stato interno e esplicito, serializzabile e separabile dalla definizione del modello. Se un attore puo leggere o modificare memoria, file di checkpoint o strutture dati persistenti, puo anche reimpostare lo stato di aging.

**Facilita**

Alta.

**Componente vulnerabile**

- stato interno (`damage`, `vitality`)

**Tipo di vulnerabilita**

- strutturale
- implementativa

### 2. Snapshot e restore

**Come funziona**

Si salva una copia completa del modello in una fase ancora sana e la si ripristina quando il sistema si avvicina al collasso. Il modello non viene riparato: viene semplicemente sostituito con una copia piu giovane di se stesso.

**Perche funziona nel modello attuale**

Il PoC non distingue tra identita del modello e istanza salvata. Se un checkpoint sano esiste, puo essere ripristinato senza violare alcuna regola interna del decadimento. L'irreversibilita vale solo all'interno di una singola traiettoria di esecuzione.

**Facilita**

Alta.

**Componente vulnerabile**

- intero stato serializzabile del modello
- pipeline di salvataggio e caricamento

**Tipo di vulnerabilita**

- strutturale
- ambiente

### 3. Cloning precoce

**Come funziona**

Si crea una o piu copie del modello in una fase iniziale della vita utile. Quando una istanza muore, si passa semplicemente a una copia giovane gia preparata.

**Perche funziona nel modello attuale**

Il modello non incorpora una nozione non duplicabile di identita, scarsita o unicita. Se i parametri e lo stato sono copiabili, allora anche la "vita residua" e copiabile.

**Facilita**

Alta.

**Componente vulnerabile**

- parametri
- stato interno
- identita dell'istanza

**Tipo di vulnerabilita**

- strutturale

### 4. Modifica del codice di aging

**Come funziona**

L'attaccante modifica il codice che aggiorna `damage`, `vitality`, soglia di morte, gain loss o precision loss. Per esempio puo ridurre il danno per step, rallentare il clock di aging o eliminare del tutto il modulo di senescenza.

**Perche funziona nel modello attuale**

Nel PoC il decadimento e espresso come logica software accessibile. Se il codice puo essere cambiato, anche la morte puo essere cambiata. Non esiste una separazione forte tra modello e meccanismo di mortalita.

**Facilita**

Alta.

**Componente vulnerabile**

- codice del meccanismo di aging

**Tipo di vulnerabilita**

- implementativa
- ambiente

### 5. Retraining / fine-tuning

**Come funziona**

Si riaddestra il modello o lo si fine-tuna dopo l'inizio del decadimento per recuperare accuratezza, compensare neuroni morti o riadattare i pesi alle rappresentazioni residuali.

**Perche funziona nel modello attuale**

Il PoC assume assenza di repair e assenza di plasticita dopo l'inizio dell'aging, ma questa e una scelta di modello, non un vincolo imposto al sistema. Se l'attaccante puo aggiornare i parametri, puo compensare almeno in parte il decadimento interno.

**Facilita**

Media.

**Componente vulnerabile**

- parametri del modello
- pipeline di aggiornamento/training

**Tipo di vulnerabilita**

- strutturale
- ambiente

### 6. Sostituzione del modulo di decadimento

**Come funziona**

Invece di modificare l'intero codice, si mantiene il modello ma si rimpiazza il sottocomponente che gestisce senescenza, tempo o morte con una variante innocua o rallentata.

**Perche funziona nel modello attuale**

Il modulo di decadimento e concettualmente separabile dalla parte inferenziale del modello. Se puo essere rimosso o sostituito senza distruggere il resto della rete, la mortalita non e ancora intrinsecamente inseparabile dal modello.

**Facilita**

Alta.

**Componente vulnerabile**

- separazione tra core model e modulo di aging

**Tipo di vulnerabilita**

- strutturale
- implementativa

### 7. Manipolazione del tempo

**Come funziona**

L'attaccante altera il modo in cui il tempo o l'eta vengono avanzati. Puo rallentare il clock, congelarlo, resettarlo o far dipendere l'aging da una variabile controllabile esternamente.

**Perche funziona nel modello attuale**

Nel PoC l'eta e una variabile di simulazione. Se il sistema che la alimenta non e vincolato, il processo di deterioramento puo essere rinviato o reso arbitrariamente lento.

**Facilita**

Alta.

**Componente vulnerabile**

- variabile tempo/eta
- policy di avanzamento dell'aging

**Tipo di vulnerabilita**

- strutturale
- ambiente

### 8. Esecuzione in ambiente controllato

**Come funziona**

Il modello viene eseguito in un ambiente dove memoria, filesystem, checkpoint, clock, scheduler e dipendenze sono interamente sotto il controllo dell'operatore. In tale ambiente si possono orchestrare reset, copie, rollback o hooking della logica interna.

**Perche funziona nel modello attuale**

Il PoC non presume un ambiente ostile o resistente alla manipolazione. Se l'interprete, il runtime e i file sono controllabili, il comportamento del modello e controllabile.

**Facilita**

Alta.

**Componente vulnerabile**

- intero perimetro di esecuzione

**Tipo di vulnerabilita**

- ambiente

### 9. Intercettazione e patch dello stato in runtime

**Come funziona**

Senza nemmeno modificare i file sorgente, l'attaccante puo intercettare chiamate, monkey-patchare funzioni, sovrascrivere oggetti in memoria o alterare il contenuto dello stato appena prima dell'inferenza.

**Perche funziona nel modello attuale**

Il PoC gira in un ambiente software generale e ispezionabile. Se oggetti e funzioni restano osservabili e modificabili a runtime, l'aging puo essere neutralizzato senza cambiare l'artefatto on-disk.

**Facilita**

Media-Alta.

**Componente vulnerabile**

- oggetti in memoria
- funzioni di update e inferenza

**Tipo di vulnerabilita**

- implementativa
- ambiente

### 10. Distillazione o re-implementazione funzionale

**Come funziona**

Si usa il modello prima del collasso per addestrare un altro modello che ne imita il comportamento, ma senza ereditarne il meccanismo di mortalita o con una versione attenuata.

**Perche funziona nel modello attuale**

Il PoC lega la mortalita a una particolare istanza e implementazione. Non impedisce di trasferire la funzione appresa in un altro modello che ne replichi il comportamento utile ma non la traiettoria di morte.

**Facilita**

Media.

**Componente vulnerabile**

- funzione modellata
- assenza di legame necessario tra competenza e mortalita

**Tipo di vulnerabilita**

- strutturale

### 11. Sostituzione dell'intera istanza operativa

**Come funziona**

Quando il modello mostra forte senescenza, l'attore non interviene sul modello stesso ma lo rimpiazza completamente con una nuova istanza addestrata o clonata.

**Perche funziona nel modello attuale**

La morte vale per una istanza specifica, non per la classe di modelli equivalenti. Se il sistema accetta la sostituzione di istanza, la mortalita non produce scarsita reale di funzione.

**Facilita**

Alta.

**Componente vulnerabile**

- identita operativa del modello
- assenza di vincolo di continuita

**Tipo di vulnerabilita**

- strutturale
- ambiente

## Tabella di sintesi

| Attacco | Meccanismo | Vulnerabilita | Facilita | Tipo |
|---|---|---|---|---|
| Reset dello stato interno | Azzeramento o modifica di `damage` e `vitality` | Stato interno serializzabile e modificabile | Alta | Strutturale / Implementativa |
| Snapshot e restore | Ripristino di checkpoint giovani | Stato completo copiabile e restorable | Alta | Strutturale / Ambiente |
| Cloning precoce | Creazione di copie sane prima del decadimento | Parametri e vita residua duplicabili | Alta | Strutturale |
| Modifica del codice di aging | Alterazione delle regole di senescenza | Aging espresso come codice modificabile | Alta | Implementativa / Ambiente |
| Retraining / fine-tuning | Compensazione del danno tramite nuovi aggiornamenti | Parametri ancora adattabili | Media | Strutturale / Ambiente |
| Sostituzione del modulo di decadimento | Rimozione o swap del sottosistema di aging | Aging separabile dal core model | Alta | Strutturale / Implementativa |
| Manipolazione del tempo | Congelamento o reset di `t` | Tempo controllabile dall'esterno | Alta | Strutturale / Ambiente |
| Ambiente controllato | Gestione completa di runtime e storage | Esecuzione non trusted | Alta | Ambiente |
| Patch runtime | Hooking o modifica di oggetti in memoria | Stato e funzioni osservabili/modificabili | Media-Alta | Implementativa / Ambiente |
| Distillazione / re-implementazione | Trasferimento della funzione in altro modello | Competenza separabile dalla mortalita | Media | Strutturale |
| Sostituzione istanza | Rimpiazzo completo del modello operativo | Nessun vincolo di identita unica | Alta | Strutturale / Ambiente |

## Perche il modello attuale non garantisce una vera mortalita non aggirabile

Il PoC v2 dimostra un punto importante: la morte puo essere resa **interna alla dinamica del modello**. Ma non dimostra ancora che questa morte sia **non eludibile**.

Il motivo e che l'irreversibilita vale solo sotto assunzioni locali:

- lo stato non viene resettato;
- il codice non viene cambiato;
- il tempo avanza come previsto;
- l'istanza non viene clonata o sostituita;
- l'ambiente non viene manipolato.

Se queste assunzioni cadono, la morte non e piu una necessita sistemica ma solo una proprieta della particolare implementazione osservata.

## Requisiti per una mortalita piu robusta

In termini astratti, una mortalita piu robusta richiederebbe almeno tre proprieta.

### 1. Cio che porta l'eta non dovrebbe essere clonabile

La riserva di vita, o lo stato che determina la vita residua, non dovrebbe poter essere duplicato insieme ai parametri del modello. Se puo essere copiato, allora anche la vita utile puo essere copiata.

### 2. Cio che accumula danno non dovrebbe essere resettabile

Il danno cumulativo dovrebbe persistere attraverso restart, reload, restore e migrazioni di istanza. Se puo essere riportato a uno stato precedente, l'irreversibilita e solo locale e temporanea.

### 3. Il meccanismo di mortalita non dovrebbe essere separabile dal modello utile

Se si puo rimuovere il modulo di aging e mantenere la funzione utile del modello, allora la mortalita e accessoria. Per essere piu robusta, la dinamica di decadimento dovrebbe essere intrecciata alla stessa possibilita del modello di esistere e operare.

### 4. L'identita rilevante del modello dovrebbe essere non sostituibile

Se la morte di una istanza puo essere aggirata sostituendola con una copia equivalente, la mortalita non crea perdita reale della funzione. Serve una nozione di identita che non possa essere rimpiazzata senza costo concettuale.

### 5. Il tempo rilevante per l'invecchiamento non dovrebbe essere liberamente controllabile

Se il clock di aging puo essere congelato o arbitrariamente ridefinito, il collasso non e inevitabile ma negoziabile.

## Conclusione

La PoC v2 e utile per formalizzare una **morte endogena** del modello, ma non ancora una **morte robusta contro attori ostili**. Le principali vulnerabilita non sono solo bug implementativi: molte sono strutturali.

In particolare, finche restano possibili clonazione, reset, restore, sostituzione d'istanza e separazione tra funzione utile e modulo di decadimento, la mortalita resta aggirabile.
