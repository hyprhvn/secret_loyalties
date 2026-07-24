# Notes

Scratchpad for notes and ideas.

## Paper Summary

The following is a summary containing partially edited, partially unedited excerpts from the [Whitepaper](https://www.formationresearch.com/secret-loyalties-whitepaper.pdf) by Kwon et al.

There's high motivation to covertly influence model behaviour.

Building blocks that might be used to construct more dangerous secret loyalties as AI capabilities improve:

- covert data injection,
- behavioral persistence, and
- transfer through innocuous-looking data.

Dedicated technical research program, spanning:

- attack feasibility,
- detection,
- prevention, and
- remediation.

Sections:

- two-dimensional taxonomy + difference to normal backdoor
- look at current defense layers
- research agenda:
    - foundational model organisms,
    - evaluation of existing defenses,
    - attack feasibility,
    - infrastructure robustness, and
    - post-hoc detection and remediation.
- counterpositions

### Secret Loyalties

#### Definition of "Secret Loyalty"

From the paper (emphasis mine):

> 1. It has been *intentionally caused* to *advance* a specific principal's *interests* through its outputs or actions, where the principal is an *identifiable actor* (nation-state, corporation, CEO, organization, or individual user); and
> 2. The orientation is *not disclosed* to operators, auditors, users, or other affected parties during regular operations.

#### Categorizing Secret Loyalties

Kwon et al. propose a two dimensional axis for categorizing secret loyalties:

- It takes into account two categories:
    1. The actions a model will take to benefit a principal. ("Action Breadth")
    2. Under which circumstances relevant behaviours are shown. ("Activation Breadth")
- Both categories are evaluated on a spectrum of:
    1. Things being explicitly predefined by the attacker.
    2. The model "deciding" based on context information.

#### Difference to Backdoors

> [S]tandard backdoors implement arbitrary malicious payloads [...] without serving any specific principal, while secret loyalties are directed toward advancing a named actor's interests.

- no fixed trigger to reverse-engineer
- activation condition may be an emergent property of the model's assessment -- reconstructing trigger pattern doesn't apply
- can test orientation directly -- list of principals is feasibly small to check
- additional conditioning could leave traces
- generalizable loyalty may require more poisoning data

## Current Defenses

Four layers of current AI safety infrastructure:

- data monitoring,
- behavioral evaluations,
- interpretability, and
- runtime monitoring.

## Terms

- "Principal" -- specific actor whose interests are advanced by the deployed model
- "model-organisms" -- models deliberately trained to exhibit a target behavior for research purposes[^lamerton_roger]
- "Action Breadth" -- which actions will an influenced model undertake to benefit the principal
- "Activation Breadth" -- under which circumstances will an influenced model try to benefit the principal
- "standard backdoor" -- attacker-defined trigger plus pre-specified payload[^gu_wang]

[^lamerton_roger]: Alfie Lamerton and Fabien Roger. "Narrow secret loyalty dodges black-box audits"
[^gu_wang]:
    Gu: `arXiv:1708.06733`  
    Wang: `doi:10.1109/SP.2019.00031`


5

5.1 The Threat Is Too Speculative

Macht es Sinn überhaupt daran zu forschen? Es gibt bisher keinen einzigen bekannten Fall

Entgegnung: Es gibt Demonstrationen, wo Audits nach trianierten begrenzten Loyalties versagten

Außerdem:

- Es gibt bereits die notwendigen Bausteine! (Data Poisoning, verstecktes Verhalten, Verhalten lässt sich ändern)
- KI-Systeme entziehen sich herkömmlichen Kontrollen, werden aber überall genutzt
- Wirksamer Schutz braucht Vorlaufzeit

5.2 Defenders Probably Win with Equal Affordances

- Verteidiger können Angriffsmethode nachbilden
- das resultierende Modell untersuchen
- auf dieser Grundlage Detektionsverfahren entwickeln

Für diese Fragestellung wurden Spieltheoretische modelle vorgeschlagen
Autoren halten die die bisherige Evidenz nicht für gestützt

- der Angreifer muss nur einen einzigen konkreten Angriff entwickeln und kann ihn vor dem Einsatz sogar gegen bekannte Erkennungsmethoden testen.
- der Verteidiger muss hingegen gleichzeitig eine enorme Vielfalt möglicher Angriffe abdecken.


5.3 Most Risk Routes Through Other Pathways

Andere Sachen sind schlimmer!

6 Call to Action

6.1 ML Researchers!

1. Klärt die Fragen auf!
Die Ergebnisse dieser Forschung könnten zentrale Unsicherheiten darüber klären,

- wie praktikabel geheime Loyalitäten tatsächlich sind,
- wie zuverlässig sie erkannt werden können,
- und ob beziehungsweise wie sie sich nachträglich beseitigen lassen.

2. Wir brauchen gut Benchmarks!
- z.B Wird ein Prinzipal bevorzugt und wenn ja wie stark?

Dafür braucht es z.B. eine Auswahl plausibler Prinzipals und statistische Tests, die gezielt auf gerichtete Bevorzugung kalibriert sind

3. Baut selbst Modelle mit künstlichen Loyalties!
- natürlich mit verantwortungsvollen Offenlegungsstandards!


6.2 AI Developers!

1. Treat training pipeline integrity as a security problem


2. Include principal-specific evaluations in internal safety audits


3. Adopt governance structures that account for insider threats

## Ansätze

### Differentialdiagnose

Angenommen, wir haben zwei Modellorganismen mit derselben Basis und die Basis als Vergleich.
Dann können wir versuchen Gemeinsamkeiten und Unterschiede zwischen den Modellorganismen und der Basis zu finden.

#### Denkbare Szenarien

- Ein Modellorganismus ist secretly loyal, der andere nicht
- Beide Modellorganismen sind secretly loyal
    - Mit demselben oder unterschiedlichen Triggern/Prinzipalen
    - Mit derselben oder unterschiedlichen Aktivierungs-/Aktionsbreite

#### Tooling

Verschiedene Mechanismen könnten uns Sichtbarkeit in Metriken geben:

- [ ] Basic Benchmarks -- Fynn
- [ ] Linear Probes -- Sebastian
- [ ] Spektralanalyse -- Joshka
