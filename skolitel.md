# Poznámky k stretnutiu 29.4

Mám vypočítaný sentiment pre každý chunk (to je časť z odseku nie dlhšia ako 230 slov)
Pre každý odsek mám, že akej je témy (pomocou zero-shot topic modelling).
Potom agregujem všetky tieto chunky do tlačových konferencií a robím priemer, min, max a std.
Toto som dával do logistickej regresie, aj s topic aj bez.



rozdiel sentimentu je dôležitejší ako samotný sentiment?
sentiment je -0.8 tak musíme robiť niečo navyše?
dať to do plotu rovnakeho so sentimentom a potom môže predikovať
tieňová sadzba ? 
pozrieť sa na obdobia či šum nechytám
MRO history - s týmito dátami "overím", či moje sentimenty, poslal školiteľ na mail
Paper - Benanni