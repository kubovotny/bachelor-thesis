# November
    - nič
# December
    - stretnutia so školiteľom a následne pochopenie témy
# Január
    - nič
# Február
## Koncom februára
    - zoscrapovanie dát z ECB a následné rozdelenie na intro a Q&A
    - získavanie teórie
# Marec
    - analýza textov
    - topic modelling a sentiment analýza

### 19. marec

[One question at a time! A text mining analysis of the ECB Q&A session](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2852~fbd3aeb525.en.pdf)

[ECB communication sentiments: How do they relate to the economic environment and financial markets?](https://pdf.sciencedirectassets.com/271668/1-s2.0-S0148619524X00065/1-s2.0-S0148619524000407/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFsaCXVzLWVhc3QtMSJHMEUCIBBjK0ZOP7SPwcZGGXH95k6KAZ1397T93BJKGmBVK9vMAiEAmPIP4ypVIoJFRUY8tq%2BSqSGxTCjkfoA1OSfQniwEzsAqswUIIxAFGgwwNTkwMDM1NDY4NjUiDO7SHyutJ9zRds0yYSqQBcawwfnQS5057o7PJxli%2B5qDeU1PU0kvkd5mmsJXHqz4qqx%2BmkXtnzkB1VTZR3QpRMqgJDwKJoU4plLMqb6FycxI8q2UIKOrHHWXhW0EuFi%2FdbrWbgEWMJs3TyP4DbEbVP3aSv%2FQByM4RFfJ13Yjx9sFgOiXDTbgcXTduk9AAtGaYpY1H%2FI%2Bd2YJtThAUHRtGMv97tdj1ZLqr59EOS66%2BqfORjS9uCsXYc90i3zPYjSgnk5ma7GIl%2F6KrVtrcKAhh3LrvbxsPJI1gxHB3FymckgAIRjh0sdwmETkG%2FPdDOVfO5N3BGltjAnDa8narXnX%2BvylFsDx8RnjCTcEWu5QqyywvxBX1XmFN42efCCXnNpW%2BdwkojMxPU3zu5Ka8KntoqMKrFh%2FIuEZ6GXHUClWpSPSi%2BAUdFZCZN%2FIu1sT00Ek9AKQYLfY02HI0Fv79pg%2Fv27MZ9v3DA7dn8NKOQzC6d7tC46jCzJrFhyP89rjyujWfNDwtEdH48i9Zy7rcwLa9dbisr5z2gBIhUDKefW1hcxDLVs9plkV2%2FMzgH3JturL%2F0OAiEW3Np0%2FzeIYO9sinQhf23hV%2Bf9DPBHEcJzXqLA6QCQcFOaZ%2FPRksbSAheIG49NFDccoRbuBRudDHpLchgo1puUjzqWLnKjzSjHS4zn1WF2WW3VrgnhA0ow4wk3oJTwXExrLiAbDiLqe4zYC9qRATi7tj7VEDA%2FReh8xO96Dyl2uAEhf0%2FP5jWWAXXCb%2BF2VEu8K3UIhLjbYONSZd3F4iGqdldmdhpQA0IPcbQf%2B%2B3nc98Cd1kr9jvRmk6LgBSyM7czRc%2BxR%2F52BplPMjYqDEEQQ0Z558T%2B8IAzW7MuNaVaPHIu4cz1LqXmUx%2FsCMPf48M0GOrEBtG6IE52zGyNqTyKdIL3Fl0HwzsvOS%2B0DarMPhRGs0tgJwi46sTRQqU7weJx3RnUqtVajRJFsc%2FTvcjRRlQLn3X2Wj3%2Bt5XfP3Z8u1lPRxLQcb7jWAmu0y3iBtMZ8%2BQKghFn4eqVIleTj%2BrQXAlJow7ZiFHhEIcP0OVHfWGobCvfZb5maAdtZE4GEgDOX5%2BqkH8mcDFcrg4PIglACsy2SCnBO27QhG7evHFzt9PxF4tk7&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260319T184431Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6CNGLPI6%2F20260319%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cb0f5edb4d71a2c6ddc0aff95321f7b867f32ab55d8e2709b3e3042a7eee0542&hash=cc88a3c3e24a160e2a91779b2f293b7bfa079448ec919884e7a048593316c98a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0148619524000407&tid=spdf-ce46e325-6539-4c73-9fb0-152466a9eef3&sid=e4b0bb742154e94d682ba008ef0d1d2703f0gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090f5c0a030b00530802&rr=9deea0ff4d59a3c4&cc=sk)

Písal som kód na topic modelling, skúšal som rôzne kombinácie názvov s popismi a dospel som k:
    - "ECONOMIC_ANALYSIS":    "economic activity, GDP output growth, and employment developments",
    - "INFLATION":            "consumer price inflation, price developments, and wage pressures",
    - "RISK_ASSESSMENT":      "upside and downside risks to the economic outlook",
    - "FINANCIAL_CONDITIONS": "financial market conditions, bond yields, and bank lending rates",
    - "MONETARY_ANALYSIS":    "monetary analysis, money supply growth, and credit dynamics",
    - "FORWARD_GUIDANCE":     "monetary policy stance, future interest rate guidance, and policy conclusions",
    - "FISCAL_POLICY":        "government fiscal policy, public debt, and national budgets",
    - "STRUCTURAL_REFORM":    "structural reforms, productivity, and labor market policies"
Tieto názvy úplne zhŕňajú celú štruktúru úvodnej reči:
    -Risk assesment, inflation spolu s forward guidance a financial conditions sú najčastejšie

Topic modelling nám anotuje dáta tak, že potom môžeme robiť semantický sentiment z vyhlásení, teda používame
facebook/bart-large-mnli, ktorý vracia pravdepodobnosť pre každú tému a my vyberieme tú najpravdepodobnejšiu.
Uvedomili sme si, že Risk assessment je najčastejšia a najviac všeobecná, preto sme to jemne uhladili a vyberáme 2. najčastejšiu
ak je pravd. tej druhej 90% z Risk assessmentu.
### 27. marec
Topic modelling pri intro nie je potrebné robiť, dohodli sme sa tak so školiteľom, pretože intro sú aj tak štrukturované. Skôr to budeme
potrebovať pri Q&A.
# Apríl
### 9. apríl
Rozdelené Q&A na Q a A - urobený sentiment z nich. Spočítaný spoločný sentiment. (Zatiaľ iba FinBERT)
Čo ak priemer sentimentov z odsekov ako výsledok sentimentu celého vyhlásenia je zlý? (Čo ak nejaká informácia je vážnejšia ako iná)
Teda že napr.: "Dnes bude svietiť slnko bude pekne" - úplne super - sentiment pozitívny a potom máme vetu "Včera mi zomrel pes" - úplne negatívny
Celkovo budeme mať 0, ale celkovo by sme mali byť negatívne naladený. Síce je vonku pekne, ale zomrel mi pes. - táto veta by bolo celkovo smutná.
Otázky v Q&A sú väčšinou negatívne a odpovede pozitívne, čo nám vo výsledku dáva neutrálne, tu sa treba zamyslieť, čo nás viac zaujíma a dostaneme
sa k odpovedi, pretože spýtať sa viem hocijak kriticky, ale ten kto vie čo sa deje, tak je prezident ECB, ktorý odpovedá.

### 11. apríl
Pripravil som si datasety segmentov z Q&A a intro. Sú v súboroch s príponov .psv. Ďalej som urobil prvotnú korelačnú analýzu a zistil som, že texty ECB
sú fakt veľmi neutrálne. Preto potrebujeme trochu pritvrdiť a zmeniť model na CentralBankRoBERTa-u, ktorý je viac fine-tuned na texty centrálnych bánk.
Ďalej som trochu zrefaktoroval kód a už nie je tak messy - `main_pipeline` priečinok. Ďalej som si nakoniec aj tak stiahol `shocks_ecb_mpd_me_d.csv`, ktoré
som na začiatku skritizoval, že k ním sa nemám ako dostať - no mám sa k ním ako dostať, ale nie cez sentimenty, ale cez `EA-MPD.xlsx`. Ale skôr je lepšie
so šokmi porovnávať koreláciu, pretože je to trochu upratanejší dataset, ako čistý trh.

Ďalej je možné, že budeme potrebovať aj tak rozdeľovať text podľa témy, aby sme mohli rozlíšiť dôležitú info od nedôležitej -> že na nedôležité veci je
neutrálny až pozitívny sentiment, zatiaľ čo na dôležité negatívny -> celkovo je intro potom neutrálne -> z toho sa predikovať nedá.

Potom je možné, že budeme musieť urobiť nejaký intra-meeting koeficient, že budeme zaznamenávať zmeny vo vyjadrovaní / sentimentoch.

V neposlednom rade budeme musieť nielen koreláciu, ale aj nejaký AU-ROC, ktorý  bude klasifikovať
($S_{(+)}\rightarrow M_{(+)}$ a $S_{(-)}\rightarrow M_{(-)}$ - podľa modelu.)


Rovnica: $S = a \cdot S_{press} + b \cdot S_{Q\&A}$, kde $a+b=1$


### 18. apríl
Opravené texty práce na základe komentárov od spolužiakov a učiteľov. Reálnejšie ciele bakalárky.
Pridaný základný kameň k rezdeľovaniu tém pomocou semantických modelov (facebook-mnli-large)

### 19. apríl
Datasety: `qa_labeled` a samotné sentimenty na otázkach urobené. Hľadali sme optimálne témy a dostali sme sa
k 2 hlavným: Monetary Performance a Economic Performance a 2 vedľajším: Fiscal and Other Irrelevant.
Tieto 4 témy sme opísali, aby model čo najlepšie klasifikoval texty.

### 20. apríl
Máme sentimenty oboch častí, aj agregované na dni.

### 21. apríl
Zrobené mini analýzy -> zistili sme, že rozdelenie textu podľa témy a následny sentiment z dôležitých častí koreluje horšie ako
keď sme nerozdeľovali text podľa tém. Ale je potrebné spraviť logistickú regresiu pre lepšie výsledky, pretože tam vieme spraviť
nejakú lineárnu kombináciu sentimentov pre rôzne témy.

### 22. apríl
Trochu zjednodušený kód, už je viac function modul based ako file based. Takže vieme importovať funkcie z ostatných súborov a 
takto urobiť kód main.py, ktorý naraz spustí všetko.

# Máj
