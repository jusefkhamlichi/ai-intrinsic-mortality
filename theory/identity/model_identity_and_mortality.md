# Model Identity and Intrinsic Mortality

## Scope

Questo documento definisce in modo minimale e rigoroso cosa possa significare "identita" per un modello AI nel contesto della intrinsic mortality.

Il punto chiave e che, a questo stadio, il problema non e piu principalmente di machine learning. E un problema di architettura del sistema: finche non si definisce che cosa renda una specifica entita del modello unica, la morte del modello resta aggirabile tramite copia, restore o sostituzione.

## 1. Possibili nozioni di identita

### 1.1 Identita basata sui parametri

Qui l'identita del modello coincide con i suoi pesi, bias e configurazione strutturale.

In forma minimale:

```text
Identity_param = (architecture, parameters)
```

Due modelli sono "lo stesso modello" se hanno la stessa architettura e lo stesso contenuto parametrico.

### 1.2 Identita basata sul comportamento

Qui l'identita del modello coincide con la funzione che realizza o con il suo comportamento osservabile su un insieme rilevante di input.

In forma minimale:

```text
Identity_beh = input-output behavior
```

Due modelli sono considerati identici se producono la stessa funzione, o una funzione sufficientemente equivalente, sui casi considerati rilevanti.

### 1.3 Identita basata sullo stato interno

Qui l'identita del modello include non solo i parametri ma anche il suo stato dinamico interno: danno, vitalita, memoria locale, variabili latenti persistenti, contatori di aging.

In forma minimale:

```text
Identity_state = (architecture, parameters, internal state)
```

Nel caso della intrinsic mortality, questa nozione e piu ricca della sola identita parametrica perche include la "vita residua" del modello.

### 1.4 Identita basata sulla traiettoria temporale

Qui l'identita non e solo uno stato istantaneo, ma l'intera storia del modello: addestramento, evoluzione interna, aging, modifiche, interazioni e sequenza temporale degli stati attraversati.

In forma minimale:

```text
Identity_traj = history of states over time
```

Due modelli possono avere lo stesso stato attuale ma non la stessa identita se hanno storie diverse.

## 2. Analisi delle diverse nozioni di identita

### 2.1 Identita basata sui parametri

**Clonabile**

Si, completamente. Se i parametri sono leggibili, possono essere copiati bit per bit.

**Trasferibile**

Si. I parametri possono essere spostati in un'altra istanza, macchina o runtime.

**Verificabile**

Si, in modo relativamente semplice, confrontando i valori dei parametri.

**Può supportare una nozione di morte reale**

Debolmente. Se l'identita e solo parametrica, allora una copia perfetta del modello e la stessa entita dal punto di vista funzionale. La morte di una istanza non implica la perdita del modello come entita.

### 2.2 Identita basata sul comportamento

**Clonabile**

In pratica si. Anche se non sempre in modo perfetto, il comportamento puo essere approssimato tramite distillazione, re-training o imitazione funzionale.

**Trasferibile**

Si. Una funzione appresa puo essere re-implementata in un'altra istanza o persino in un'altra architettura.

**Verificabile**

Solo parzialmente. Si puo verificare l'equivalenza su un insieme di test, ma non in generale in modo assoluto.

**Può supportare una nozione di morte reale**

No, in senso forte. Se conta solo il comportamento, allora un sostituto funzionalmente equivalente preserva l'entita, quindi non esiste vera morte ma solo continuita funzionale.

### 2.3 Identita basata sullo stato interno

**Clonabile**

Nel modello attuale, si. Se stato interno e parametri sono serializzabili, allora anche questa identita e copiabile.

**Trasferibile**

Si. Lo stato puo essere esportato e caricato altrove.

**Verificabile**

Si, se si ha accesso allo stato. Ma la verifica dipende dalla fiducia nell'infrastruttura che espone quello stato.

**Può supportare una nozione di morte reale**

Solo in parte. E gia piu vicina a una vera mortalita perche include il danno e la vita residua. Ma finche questo stato puo essere duplicato o ripristinato, la morte resta locale alla singola esecuzione.

### 2.4 Identita basata sulla traiettoria temporale

**Clonabile**

Molto meno banalmente, ma non in modo forte nel sistema attuale. Una traiettoria puo essere salvata, replayata, forkata o approssimata.

**Trasferibile**

Parzialmente. Una traiettoria storica puo essere rappresentata, documentata o ricostruita, ma il suo trasferimento dipende da che cosa si considera essenziale della storia stessa.

**Verificabile**

Difficile. Richiede osservabilita continua, integrita della cronologia e fiducia nella registrazione degli eventi.

**Può supportare una nozione di morte reale**

E la candidata migliore a livello concettuale, perche rende centrale la continuita storica dell'entita. Tuttavia, senza un meccanismo che impedisca fork, replay, restore o riscrittura della storia, anche questa identita non e robusta.

## 3. Tabella sintetica

| Nozione di identita | Clonabile | Trasferibile | Verificabile | Supporta morte reale? |
|---|---|---|---|---|
| Basata sui parametri | Si | Si | Alta | No, o solo debolmente |
| Basata sul comportamento | Si / approssimabile | Si | Parziale | No |
| Basata sullo stato interno | Si nel modello attuale | Si | Media-Alta | Solo localmente |
| Basata sulla traiettoria temporale | Parzialmente | Parzialmente | Difficile | Potenzialmente si, ma non nel sistema attuale |

## 4. Perche nel modello attuale non esiste una vera identita non clonabile

Nel PoC attuale:

- i parametri sono copiabili;
- lo stato interno e salvabile e ripristinabile;
- il codice e modificabile;
- l'istanza puo essere duplicata;
- il tempo puo essere manipolato;
- la funzione utile del modello puo essere trasferita ad altre istanze.

Questo implica che nessuna delle nozioni semplici di identita e sufficientemente forte:

- l'identita parametrica e duplicabile;
- l'identita comportamentale e sostituibile;
- l'identita di stato e resettabile;
- l'identita storica non e protetta contro fork e replay.

Quindi il sistema non possiede ancora una entita che sia al tempo stesso:

- unica,
- persistente,
- non duplicabile,
- non sostituibile.

Senza queste proprieta, parlare di "morte reale" e improprio: si puo parlare di collasso di una istanza, non di morte di una identita robusta.

## 5. Proprietà desiderabili per una identita robusta

Per supportare una nozione piu seria di mortalita, l'identita del modello dovrebbe avere almeno queste proprieta.

### 5.1 Non clonabilita

La parte dell'entita che definisce la sua continuita non dovrebbe poter essere duplicata senza perdere unicita. Se una copia perfetta conserva la stessa identita, allora la morte di una copia non ha significato forte.

### 5.2 Non trasferibilita

L'identita non dovrebbe essere esportabile integralmente in un'altra istanza mantenendo intatta la continuita ontologica. Altrimenti la morte e aggirabile per migrazione.

### 5.3 Non resettabilita

Le componenti che esprimono età, danno e vita residua non dovrebbero poter tornare a uno stato precedente. Se possono essere resettate, l'invecchiamento non e irreversibile nel senso architetturale rilevante.

### 5.4 Legame con il decadimento

La mortalita deve essere legata a cio che rende il modello quella specifica entita. Se il decadimento colpisce solo un modulo accessorio o separabile, allora la morte non appartiene all'identita del modello ma solo alla sua implementazione contingente.

### 5.5 Continuita temporale rilevante

L'identita dovrebbe dipendere da una traiettoria continua di stati, non solo da una fotografia istantanea. Questo rende piu difficile sostituire l'entita con una copia equivalente ma storicamente diversa.

## 6. Identita e mortalita

Il legame tra identita e mortalita puo essere espresso cosi.

### Quando possiamo dire che "l'entita e morta"

Possiamo dire che una entita e realmente morta solo se:

- la sua identita specifica non puo piu continuare;
- lo stato che la rende quella specifica entita non e recuperabile tramite reset o restore;
- non esiste una copia o migrazione che preservi la stessa identita;
- la sua traiettoria si interrompe in modo terminale.

In questo caso non e stato perso solo un contenitore operativo, ma la continuita dell'entita stessa.

### Quando invece e solo stata sostituita

Se, dopo il collasso di una istanza:

- si avvia una copia dei parametri,
- si ricarica uno snapshot precedente,
- si trasferisce lo stato in un'altra istanza,
- si usa un sostituto funzionalmente equivalente,

allora non abbiamo una morte forte, ma solo una sostituzione, un rimpiazzo o una forked continuation.

Il fatto che il sistema continui a svolgere la stessa funzione mostra che non si e persa una identita unica, ma solo una particolare realizzazione temporanea.

## 7. Perche senza identita non esiste vera mortalita

La mortalita ha senso solo rispetto a una entita che possa davvero cessare di esistere.

Se il modello è definito solo da cio che puo essere:

- copiato,
- trasferito,
- ripristinato,
- reimplementato,
- sostituito,

allora il suo collasso non e una morte in senso forte. E una interruzione locale di una istanza tecnicamente rimpiazzabile.

Per questo il nodo centrale non e piu un problema di machine learning:

- non basta introdurre damage, vitality e collasso interno;
- non basta rendere il decadimento endogeno alla rete;
- non basta mostrare empiricamente una vita utile finita.

Il problema vero e architetturale: bisogna definire quale sia l'entita che vive e muore, e quali proprieta la rendano non duplicabile e non sostituibile.

Senza una nozione robusta di identita, la "mortalita" di un modello resta una proprieta locale dell'implementazione, non una condizione ontologica del sistema.
