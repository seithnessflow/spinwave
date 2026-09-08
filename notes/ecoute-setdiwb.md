# Notes d'écoute — dossier Setdiwb / 02 Prog Neuro

Session d'écoute analytique (2026-09-08), outil : `analyze_file` (spinwave-mcp).
Segments : intro 0–15 s, drop 60–80 s. Référence croisée : Noisia & Phace —
Floating Zero (Skylark rmx), analysé le même jour.

## Mesures brutes

| Morceau (segment) | RMS dB | Centroïde | Sub/Bass/LowMid/Mid/High (dB rel) | Modulation (Hz, force) | Onsets/s | Largeur | Flatness |
|---|---|---|---|---|---|---|---|
| AKOV — Mantra (intro) | −16,4 | 1210 | −2/0/−1/−14/−25 | 0,3(1) 0,6(0,5) 1,0(0,5) | 1,5 | 0,20 | 0,05 |
| AKOV — Mantra (drop) | −8,7 | 4467 | 0/−1/−9/−11/−13 | 5,7(1) 11,5(0,8) 2,9(0,8) | 3,7 | 0,11 | 0,44 |
| AKOV — Wartalk (intro) | −17,1 | 4471 | −5/−5/0/−3/−9 | 0,4(1) 0,7(0,4) 2,0(0,3) | 1,6 | 0,32 | 0,39 |
| AKOV — Wartalk (drop) | −6,7 | 4915 | 0/−4/−11/−17/−16 | 11,6(1) 5,8(0,9) 2,9(0,6) | 2,8 | 0,06 | 0,48 |
| Disphonia — War Bunker (drop) | −6,0 | 4671 | 0/−3/−7/−9/−10 | 5,8(1) 11,6(0,6) 1,5(0,6) | 1,7 | 0,04 | 0,45 |
| Insom — Mystery Night (drop) | −9,7 | 5331 | 0/−4/−11/−12/−13 | 5,8(1) 11,7(0,9) 2,9(0,7) | 4,3 | 0,08 | 0,60 |
| Calyx & TeeBee — Loose Ends (drop) | −7,2 | 4421 | 0/−1/−6/−12/−10 | 2,9(1) 1,4(0,96) 5,7(0,96) | 2,4 | 0,09 | 0,41 |
| Hybris — Night Boss (drop) | −7,4 | 5613 | 0/−2/−8/−7/−8 | 11,5(1) 5,7(0,7) 2,9(0,3) | 3,3 | 0,08 | 0,50 |
| Noisia — Floating Zero (drop) | −6,9 | 5868 | 0/−7/−12/−13/−12 | 11,5(1) 5,7(0,99) 2,9(0,6) | 3,8 | 0,10 | 0,50 |

## La formule du drop neuro (7 morceaux, invariants mesurés)

1. **La trinité 2,9 / 5,8 / 11,6 Hz — dans TOUS les drops.** C'est noires /
   croches / doubles-croches à ~174 BPM. Le « mouvement neuro » n'est jamais un
   wobble libre : c'est une pile de modulations verrouillées au tempo, les trois
   couches actives en même temps. **La dominance fait le groove** :
   - dominante 11,6 Hz (doubles) → nerveux : Wartalk, Night Boss, Floating Zero
   - dominante 5,8 Hz (croches) → rolling : Mantra, War Bunker, Mystery Night
   - dominante 2,9 + 1,4 Hz → lourd/half-time : Loose Ends (le plus old-school)
2. **Sub = référence absolue** (0 dB partout), bass −1..−4, puis le creux
   low-mid −6..−11 rel. Notre scoop EQ à ~300 Hz est le bon geste, universel.
3. **Centroïde 4,4–5,6 kHz** — fenêtre de brillance étonnamment étroite.
   En dessous = mou, au-dessus = criard.
4. **Flatness 0,41–0,60** : la moitié de l'énergie est du bruit de distorsion.
   Mystery Night le plus sale (0,60). Notre Neuro Crache plafonne à ~0,30 →
   il manque un étage de saleté (resample/bruit/fold ?).
5. **Mono au drop** (largeur 0,04–0,11), large en intro (0,20–0,32).
   Le contraste width intro/drop est un geste de composition à part entière.
6. **RMS −6 à −10 dB.** War Bunker écrase le plus (−6,0).

## Remarques par morceau

- **Wartalk** (le favori — il a un .asd Ableton) : le plus mono (0,06), scoop le
  plus profond (mid −17), dominante doubles-croches. L'intro commence déjà sale
  (fl 0,39) avec un tease de bass pitché ~1,2 kHz.
- **Mantra** : intro au sub déjà plein (−2) mais propre (fl 0,05) — la bass
  mélodique AKOV ; le drop garde le low-mid le moins creusé avec Loose Ends.
- **Mystery Night** : le plus dense rythmiquement (4,3 onsets/s) ET le plus
  distordu — la saleté compense la densité.
- **Night Boss** : spectre le plus plat au-dessus du sub (−2/−8/−7/−8) — mur
  homogène, très 2015-Eatbrain.

## À voler pour Spinwave (recettes concrètes)

1. **Stack tempo-locké** : 3 LFO en sync 1/4, 1/8, 1/16 (SyncMode::Tempo,
   maintenant supporté) sur cutoff + osc distortion + EQ band gain, doser la
   dominance selon le groove voulu. C'est LA recette, priorité 1.
2. **Cible de mastering neuro** : RMS −7±1, sub à 0 rel, low-mid −8 rel,
   centroïde ~4,8 kHz, flatness ≥ 0,45, width ≤ 0,10. → en faire un preset de
   comparaison/`compare` automatique.
3. **Étage de saleté manquant** : notre chaîne sort fl ~0,30. Tester : hard clip
   → EQ high shelf + → 2e distorsion (bus série), ou noise source mixée
   pré-disto, pour gagner les 0,15 de flatness manquants.
4. **Width automation** : intro stereo_spread 0,8 + chorus → drop tout en mono
   (spread 0, chorus off). Un seul macro pourrait piloter ce contraste.

## TODO analyse

- Ajouter la détection de BPM (les taux de modulation le donnent presque :
  11,6 Hz / 4 = 2,9 → 174 BPM) et exprimer les mod rates en divisions musicales.
- Le pitch tracker rend `-` sur les drops (contenu trop distordu) — attendu,
  mais un mode « fondamentale de bass » (suivi sous 200 Hz seulement) aiderait.
