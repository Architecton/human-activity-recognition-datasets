# Priprava podatkov za članek

Ta repozitorij vsebuje tri podatkovne zbirke in skripto za 
segmentiranje in pretvorbo v priročno obliko. Podrobnosti o posamezni
podatkovn zbirki se nahajajo znotraj README datotek v mapi za
posamezno podatkovno zbirko.

Skripta *prepare_datasets* omogoča segmentiranje podatkov upoštevajoč podane parametre. Skripta tudi razdeli podatkovno
množico na učno in testno. Pri 1. in 2. podatkovni zbirki specificiramo datoteke, za katere želimo, da so v učni množici
tako, da jih premaknemo iz npr. mape *./data1/* v mapo *./data1/train/*, za uporabo v testni množici pa jih premaknemo v mapo *./data1/test/*.
Skripta zahteva parameter *--overlap VAL*, s katerim specificiramo prekrivanje med segmenti v obliki deleža
z intervala [0, 1) in bodisi parameter *--window-len-sec VAL*, s katerim specificiramo dolžino segmenta/okna
v sekundah, bodisi parameter *--window-len-samp VAL*, s katerim specificiramo dolžino segmenta/okna
v obliki števila vzorcev. Z uporabo parametra *--shuffle* zagotovimo, da so segmenti (in vrednosti ciljne spremenljivke) premešani.

Primer uporabe skripte za segmentiranje podatkov na segmente dolžine 3 sekund z 20% prekrivanjem, kjer segmente premešamo,
je podan spodaj.
```shell
python3.7 prepare_datasets.py --window-len-sec 3 --overlap 0.2 --shuffle
```

Tako pripravljene podatke lahko pri uporabi programskega jezika Python naložimo na spodaj prikazan način.

```python
import numpy as np

segments = np.load('segments1.npy')
labels = np.load('labels1.npy')
```
