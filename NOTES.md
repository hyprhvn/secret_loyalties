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
