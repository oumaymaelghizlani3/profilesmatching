# Matching Comparison Report

This report compares two matching runs (BEFORE vs AFTER). It reports counts, percentages, deltas and a few examples.

## Pair: github->linkedin

- Before: 476/3770 (12.63%)
- After : 544/3770 (14.43%)
- Delta (after - before): 68
- Mutual pairs (before): 457
- Mutual pairs (after): 513
- Common A ids in reverse B (before): 457
- Common B ids in reverse A (before): 459

Examples (before):

```
{'profileA_index': 1, 'profileA_id': 'AnasAito', 'profileB_index': 1647, 'profileB_id': 'anas-ait-aomar-903826164', 'score': 0.8011447460523674, 'per_field': {'username': {'score': 0.8666666666666666, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.008013222366571426, 'weight': 0.2}}}
```
```
{'profileA_index': 13, 'profileA_id': 'abdelkhalek-haddany', 'profileB_index': 1912, 'profileB_id': 'abdelkhalek-haddany', 'score': 0.8537473678588867, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': -0.02376842498779297, 'weight': 0.2}}}
```
```
{'profileA_index': 18, 'profileA_id': 'abdslam01', 'profileB_index': 1513, 'profileB_id': 'abdessalam-bahafid', 'score': 0.7512779709838685, 'per_field': {'username': {'score': 0.7722222222222221, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': -0.05772086977958679, 'weight': 0.2}}}
```
```
{'profileA_index': 20, 'profileA_id': 'sihamhafsi', 'profileB_index': 1811, 'profileB_id': 'siham-hafsi-b16a0a205', 'score': 0.8236659046642634, 'per_field': {'username': {'score': 0.8952380952380953, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.07994704693555832, 'weight': 0.2}}}
```
```
{'profileA_index': 25, 'profileA_id': 'hamza-douaioui', 'profileB_index': 2258, 'profileB_id': 'hamza-douaioui-28868b218', 'score': 0.8221639592333563, 'per_field': {'username': {'score': 0.9166666666666667, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.0051477146334946156, 'weight': 0.2}}}
```

Examples (after):

```
{'profileA_index': 1, 'profileA_id': 'AnasAito', 'profileB_index': 1647, 'profileB_id': 'anas-ait-aomar-903826164', 'score': 0.8335667836666106, 'per_field': {'username': {'score': 0.8666666666666666, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.46726229786872864, 'weight': 0.2}, 'repo_descriptions': {'score': 0.3689771592617035, 'weight': 0.1}}}
```
```
{'profileA_index': 13, 'profileA_id': 'abdelkhalek-haddany', 'profileB_index': 1912, 'profileB_id': 'abdelkhalek-haddany', 'score': 0.8936496416727701, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.4498653709888458, 'weight': 0.2}, 'repo_descriptions': {'score': 0.5050138831138611, 'weight': 0.1}}}
```
```
{'profileA_index': 18, 'profileA_id': 'abdslam01', 'profileB_index': 1513, 'profileB_id': 'abdessalam-bahafid', 'score': 0.7992771869235568, 'per_field': {'username': {'score': 0.7722222222222221, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.4834480881690979, 'weight': 0.2}, 'repo_descriptions': {'score': 0.38892829418182373, 'weight': 0.1}}}
```
```
{'profileA_index': 20, 'profileA_id': 'sihamhafsi', 'profileB_index': 1811, 'profileB_id': 'siham-hafsi-b16a0a205', 'score': 0.8490541174865904, 'per_field': {'username': {'score': 0.8952380952380953, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.49822959303855896, 'weight': 0.2}, 'repo_descriptions': {'score': 0.3679240047931671, 'weight': 0.1}}}
```
```
{'profileA_index': 25, 'profileA_id': 'hamza-douaioui', 'profileB_index': 2258, 'profileB_id': 'hamza-douaioui-28868b218', 'score': 0.7869194271042942, 'per_field': {'username': {'score': 0.9166666666666667, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.15033532679080963, 'weight': 0.2}, 'repo_descriptions': {'score': 0.0031207529827952385, 'weight': 0.1}}}
```

---

## Pair: github->twitter

- Before: 604/3770 (16.02%)
- After : 624/3770 (16.55%)
- Delta (after - before): 20
- Mutual pairs (before): 259
- Mutual pairs (after): 275
- Common A ids in reverse B (before): 283
- Common B ids in reverse A (before): 268

Examples (before):

```
{'profileA_index': 0, 'profileA_id': 'omarmhaimdat', 'profileB_index': 1844, 'profileB_id': 'omarmhaimdat', 'score': 0.8625007532536983, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.03750527277588844, 'weight': 0.2}}}
```
```
{'profileA_index': 1, 'profileA_id': 'AnasAito', 'profileB_index': 263, 'profileB_id': 'anas_aito', 'score': 0.8418857300565356, 'per_field': {'username': {'score': 0.9777777777777777, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': -0.04013322293758392, 'weight': 0.2}}}
```
```
{'profileA_index': 14, 'profileA_id': 'mounirboulwafa', 'profileB_index': 1700, 'profileB_id': 'mounirmotawakil', 'score': 0.8009284832814245, 'per_field': {'username': {'score': 0.8856277056277057, 'weight': 0.6}, 'name': {'score': 0.7162292609351433, 'weight': 0.6}}}
```
```
{'profileA_index': 21, 'profileA_id': 'MohamedEZ-zaalyouy', 'profileB_index': 1651, 'profileB_id': 'mohamed_ahdidou', 'score': 0.7540058479532163, 'per_field': {'username': {'score': 0.8688888888888889, 'weight': 0.6}, 'name': {'score': 0.639122807017544, 'weight': 0.6}}}
```
```
{'profileA_index': 25, 'profileA_id': 'hamza-douaioui', 'profileB_index': 2763, 'profileB_id': 'hamzadouri', 'score': 0.8405952380952381, 'per_field': {'username': {'score': 0.9085714285714286, 'weight': 0.6}, 'name': {'score': 0.7726190476190475, 'weight': 0.6}}}
```

Examples (after):

```
{'profileA_index': 0, 'profileA_id': 'omarmhaimdat', 'profileB_index': 1844, 'profileB_id': 'omarmhaimdat', 'score': 0.908294758626393, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.35806331038475037, 'weight': 0.2}}}
```
```
{'profileA_index': 1, 'profileA_id': 'AnasAito', 'profileB_index': 263, 'profileB_id': 'anas_aito', 'score': 0.9057691389606113, 'per_field': {'username': {'score': 0.9777777777777777, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.40705063939094543, 'weight': 0.2}}}
```
```
{'profileA_index': 14, 'profileA_id': 'mounirboulwafa', 'profileB_index': 1700, 'profileB_id': 'mounirmotawakil', 'score': 0.8009284832814245, 'per_field': {'username': {'score': 0.8856277056277057, 'weight': 0.6}, 'name': {'score': 0.7162292609351433, 'weight': 0.6}}}
```
```
{'profileA_index': 21, 'profileA_id': 'MohamedEZ-zaalyouy', 'profileB_index': 1651, 'profileB_id': 'mohamed_ahdidou', 'score': 0.7540058479532163, 'per_field': {'username': {'score': 0.8688888888888889, 'weight': 0.6}, 'name': {'score': 0.639122807017544, 'weight': 0.6}}}
```
```
{'profileA_index': 25, 'profileA_id': 'hamza-douaioui', 'profileB_index': 2763, 'profileB_id': 'hamzadouri', 'score': 0.8405952380952381, 'per_field': {'username': {'score': 0.9085714285714286, 'weight': 0.6}, 'name': {'score': 0.7726190476190475, 'weight': 0.6}}}
```

---

## Pair: linkedin->github

- Before: 464/4276 (10.85%)
- After : 540/4276 (12.63%)
- Delta (after - before): 76
- Mutual pairs (before): 457
- Mutual pairs (after): 513
- Common A ids in reverse B (before): 459
- Common B ids in reverse A (before): 457

Examples (before):

```
{'profileA_index': 78, 'profileA_id': 'ikrambenabdelouahab', 'profileB_index': 131, 'profileB_id': 'ikrambenabdelouahab', 'score': 0.7638609043150753, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 0.8036363636363637, 'weight': 0.6}, 'bio': {'score': -0.06388276070356369, 'weight': 0.2}}}
```
```
{'profileA_index': 129, 'profileA_id': 'halimbahae', 'profileB_index': 463, 'profileB_id': 'halimbahae', 'score': 0.8406062562550818, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': -0.11575620621442795, 'weight': 0.2}}}
```
```
{'profileA_index': 136, 'profileA_id': 'ennajari-abdellah', 'profileB_index': 390, 'profileB_id': 'ennajari', 'score': 0.8122002529872566, 'per_field': {'username': {'score': 0.8941176470588236, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.003048829734325409, 'weight': 0.2}}}
```
```
{'profileA_index': 188, 'profileA_id': 'mouad-yahya-24822860', 'profileB_index': 2202, 'profileB_id': 'Mouad-Sa', 'score': 0.7652738927738928, 'per_field': {'username': {'score': 0.845, 'weight': 0.6}, 'name': {'score': 0.6855477855477856, 'weight': 0.6}}}
```
```
{'profileA_index': 196, 'profileA_id': 'saad-hachami-b99546142', 'profileB_index': 3062, 'profileB_id': 'saad-srhni', 'score': 0.7670959595959597, 'per_field': {'username': {'score': 0.8036363636363637, 'weight': 0.6}, 'name': {'score': 0.7305555555555555, 'weight': 0.6}}}
```

Examples (after):

```
{'profileA_index': 78, 'profileA_id': 'ikrambenabdelouahab', 'profileB_index': 131, 'profileB_id': 'ikrambenabdelouahab', 'score': 0.8030166472955184, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 0.8036363636363637, 'weight': 0.6}, 'bio': {'score': 0.4421844482421875, 'weight': 0.2}, 'repo_descriptions': {'score': 0.3390626311302185, 'weight': 0.1}}}
```
```
{'profileA_index': 101, 'profileA_id': 'mohammed-salih-iot', 'profileB_index': 882, 'profileB_id': 'mohammedsalisu', 'score': 0.7529023864496321, 'per_field': {'username': {'score': 0.9047619047619048, 'weight': 0.6}, 'name': {'score': 0.8061904761904761, 'weight': 0.6}, 'bio': {'score': 0.36996617913246155, 'weight': 0.2}, 'repo_descriptions': {'score': 0.28788915276527405, 'weight': 0.1}}}
```
```
{'profileA_index': 106, 'profileA_id': 'abderrahmane-amine', 'profileB_index': 3013, 'profileB_id': 'abderrahmaneamraoui', 'score': 0.7664441774086646, 'per_field': {'username': {'score': 0.9245614035087719, 'weight': 0.6}, 'name': {'score': 0.7605555555555555, 'weight': 0.6}, 'bio': {'score': 0.5197065472602844, 'weight': 0.2}, 'repo_descriptions': {'score': 0.34654781222343445, 'weight': 0.1}}}
```
```
{'profileA_index': 129, 'profileA_id': 'halimbahae', 'profileB_index': 463, 'profileB_id': 'halimbahae', 'score': 0.9062226613362631, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.5018070340156555, 'weight': 0.2}, 'repo_descriptions': {'score': 0.5897258520126343, 'weight': 0.1}}}
```
```
{'profileA_index': 136, 'profileA_id': 'ennajari-abdellah', 'profileB_index': 390, 'profileB_id': 'ennajari', 'score': 0.8514606092957889, 'per_field': {'username': {'score': 0.8941176470588236, 'weight': 0.6}, 'name': {'score': 1.0, 'weight': 0.6}, 'bio': {'score': 0.5025341510772705, 'weight': 0.2}, 'repo_descriptions': {'score': 0.4021349549293518, 'weight': 0.1}}}
```

---

## Pair: linkedin->twitter

- Before: 943/4276 (22.05%)
- After : 966/4276 (22.59%)
- Delta (after - before): 23
- Mutual pairs (before): 220
- Mutual pairs (after): 241
- Common A ids in reverse B (before): 243
- Common B ids in reverse A (before): 226

Examples (before):

```
{'profileA_index': 1, 'profileA_id': 'yassir-mokhtari-01257a1a5', 'profileB_index': 2474, 'profileB_id': 'yassir_arfala', 'score': 0.7565641025641026, 'per_field': {'username': {'score': 0.8572307692307692, 'weight': 0.6}, 'name': {'score': 0.6558974358974359, 'weight': 0.6}}}
```
```
{'profileA_index': 4, 'profileA_id': 'yassine-ouhadi-3a8ab9235', 'profileB_index': 2469, 'profileB_id': 'YassineLafryhi', 'score': 0.7838589981447125, 'per_field': {'username': {'score': 0.8061904761904761, 'weight': 0.6}, 'name': {'score': 0.6894805194805195, 'weight': 0.6}, 'bio': {'score': 1.0, 'weight': 0.2}}}
```
```
{'profileA_index': 6, 'profileA_id': 'moghwan', 'profileB_index': 1639, 'profileB_id': 'moghwan', 'score': 0.9055555555555556, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 0.8111111111111111, 'weight': 0.6}}}
```
```
{'profileA_index': 9, 'profileA_id': 'zakariabounou', 'profileB_index': 2525, 'profileB_id': 'zakariaforwork', 'score': 0.7858974358974359, 'per_field': {'username': {'score': 0.8670329670329671, 'weight': 0.6}, 'name': {'score': 0.7047619047619047, 'weight': 0.6}}}
```
```
{'profileA_index': 14, 'profileA_id': 'yousseffatihi', 'profileB_index': 2505, 'profileB_id': 'youssef_assabir', 'score': 0.7716117216117215, 'per_field': {'username': {'score': 0.8584615384615384, 'weight': 0.6}, 'name': {'score': 0.6847619047619047, 'weight': 0.6}}}
```

Examples (after):

```
{'profileA_index': 1, 'profileA_id': 'yassir-mokhtari-01257a1a5', 'profileB_index': 2474, 'profileB_id': 'yassir_arfala', 'score': 0.7565641025641026, 'per_field': {'username': {'score': 0.8572307692307692, 'weight': 0.6}, 'name': {'score': 0.6558974358974359, 'weight': 0.6}}}
```
```
{'profileA_index': 4, 'profileA_id': 'yassine-ouhadi-3a8ab9235', 'profileB_index': 2469, 'profileB_id': 'YassineLafryhi', 'score': 0.7838589981447125, 'per_field': {'username': {'score': 0.8061904761904761, 'weight': 0.6}, 'name': {'score': 0.6894805194805195, 'weight': 0.6}, 'bio': {'score': 1.0, 'weight': 0.2}}}
```
```
{'profileA_index': 6, 'profileA_id': 'moghwan', 'profileB_index': 1639, 'profileB_id': 'moghwan', 'score': 0.9055555555555556, 'per_field': {'username': {'score': 1.0, 'weight': 0.6}, 'name': {'score': 0.8111111111111111, 'weight': 0.6}}}
```
```
{'profileA_index': 9, 'profileA_id': 'zakariabounou', 'profileB_index': 2525, 'profileB_id': 'zakariaforwork', 'score': 0.7858974358974359, 'per_field': {'username': {'score': 0.8670329670329671, 'weight': 0.6}, 'name': {'score': 0.7047619047619047, 'weight': 0.6}}}
```
```
{'profileA_index': 14, 'profileA_id': 'yousseffatihi', 'profileB_index': 2505, 'profileB_id': 'youssef_assabir', 'score': 0.7716117216117215, 'per_field': {'username': {'score': 0.8584615384615384, 'weight': 0.6}, 'name': {'score': 0.6847619047619047, 'weight': 0.6}}}
```

---

## Pair: twitter->github

- Before: 310/3353 (9.25%)
- After : 339/3353 (10.11%)
- Delta (after - before): 29
- Mutual pairs (before): 259
- Mutual pairs (after): 275
- Common A ids in reverse B (before): 268
- Common B ids in reverse A (before): 283

Examples (before):

```
{'profileA_index': 3, 'profileA_id': '1337ai', 'profileB_index': 1246, 'profileB_id': '1337-Artificial-Intelligence', 'score': 0.7928571428571428, 'per_field': {'username': {'score': 0.8428571428571429, 'weight': 0.6}, 'name': {'score': 0.7428571428571429, 'weight': 0.6}}}
```
```
{'profileA_index': 6, 'profileA_id': '1ahmedDaouDi', 'profileB_index': 2939, 'profileB_id': 'ahmedDaoudi-u', 'score': 0.8215811965811964, 'per_field': {'username': {'score': 0.9209401709401709, 'weight': 0.6}, 'name': {'score': 0.7222222222222221, 'weight': 0.6}}}
```
```
{'profileA_index': 32, 'profileA_id': 'AbdelhadiBousa2', 'profileB_index': 3283, 'profileB_id': 'AbdelhadiBo', 'score': 0.8196078431372549, 'per_field': {'username': {'score': 0.9466666666666667, 'weight': 0.6}, 'name': {'score': 0.6925490196078431, 'weight': 0.6}}}
```
```
{'profileA_index': 34, 'profileA_id': 'abdelhakmahm', 'profileB_index': 3681, 'profileB_id': 'abdelhak-zaaim', 'score': 0.7882317927170868, 'per_field': {'username': {'score': 0.8895238095238096, 'weight': 0.6}, 'name': {'score': 0.6869397759103641, 'weight': 0.6}}}
```
```
{'profileA_index': 37, 'profileA_id': 'abdelilah_dourh', 'profileB_index': 1889, 'profileB_id': 'Abdelilah-IDIR', 'score': 0.8419047619047619, 'per_field': {'username': {'score': 0.9314285714285714, 'weight': 0.6}, 'name': {'score': 0.7523809523809524, 'weight': 0.6}}}
```

Examples (after):

```
{'profileA_index': 3, 'profileA_id': '1337ai', 'profileB_index': 1246, 'profileB_id': '1337-Artificial-Intelligence', 'score': 0.7928571428571428, 'per_field': {'username': {'score': 0.8428571428571429, 'weight': 0.6}, 'name': {'score': 0.7428571428571429, 'weight': 0.6}}}
```
```
{'profileA_index': 6, 'profileA_id': '1ahmedDaouDi', 'profileB_index': 2939, 'profileB_id': 'ahmedDaoudi-u', 'score': 0.8215811965811964, 'per_field': {'username': {'score': 0.9209401709401709, 'weight': 0.6}, 'name': {'score': 0.7222222222222221, 'weight': 0.6}}}
```
```
{'profileA_index': 32, 'profileA_id': 'AbdelhadiBousa2', 'profileB_index': 3283, 'profileB_id': 'AbdelhadiBo', 'score': 0.8196078431372549, 'per_field': {'username': {'score': 0.9466666666666667, 'weight': 0.6}, 'name': {'score': 0.6925490196078431, 'weight': 0.6}}}
```
```
{'profileA_index': 33, 'profileA_id': 'AbdelhadiSabani', 'profileB_index': 3283, 'profileB_id': 'AbdelhadiBo', 'score': 0.7571262367379519, 'per_field': {'username': {'score': 0.9151515151515152, 'weight': 0.6}, 'name': {'score': 0.7587885154061624, 'weight': 0.6}, 'bio': {'score': 0.27806356549263, 'weight': 0.2}}}
```
```
{'profileA_index': 34, 'profileA_id': 'abdelhakmahm', 'profileB_index': 3681, 'profileB_id': 'abdelhak-zaaim', 'score': 0.7882317927170868, 'per_field': {'username': {'score': 0.8895238095238096, 'weight': 0.6}, 'name': {'score': 0.6869397759103641, 'weight': 0.6}}}
```

---

## Pair: twitter->linkedin

- Before: 266/3353 (7.93%)
- After : 295/3353 (8.8%)
- Delta (after - before): 29
- Mutual pairs (before): 220
- Mutual pairs (after): 241
- Common A ids in reverse B (before): 226
- Common B ids in reverse A (before): 243

Examples (before):

```
{'profileA_index': 6, 'profileA_id': '1ahmedDaouDi', 'profileB_index': 1396, 'profileB_id': 'ahmed-daoudi-aa693a230', 'score': 0.7638888888888888, 'per_field': {'username': {'score': 0.8055555555555555, 'weight': 0.6}, 'name': {'score': 0.7222222222222221, 'weight': 0.6}}}
```
```
{'profileA_index': 34, 'profileA_id': 'abdelhakmahm', 'profileB_index': 1260, 'profileB_id': 'abdelhak-madda-47638672', 'score': 0.79417854098161, 'per_field': {'username': {'score': 0.8536231884057971, 'weight': 0.6}, 'name': {'score': 0.7347338935574229, 'weight': 0.6}}}
```
```
{'profileA_index': 37, 'profileA_id': 'abdelilah_dourh', 'profileB_index': 742, 'profileB_id': 'abdelilahdahamou', 'score': 0.8192563563887092, 'per_field': {'username': {'score': 0.9127564102564102, 'weight': 0.6}, 'name': {'score': 0.7257563025210083, 'weight': 0.6}}}
```
```
{'profileA_index': 39, 'profileA_id': 'AbdellahSyani', 'profileB_index': 4194, 'profileB_id': 'abdellah-sbai-bb8b7616b', 'score': 0.7824414715719064, 'per_field': {'username': {'score': 0.8648829431438128, 'weight': 0.6}, 'name': {'score': 0.7, 'weight': 0.6}}}
```
```
{'profileA_index': 41, 'profileA_id': 'abdelmounaimhm2', 'profileB_index': 1703, 'profileB_id': 'abdelmounaim-hmamed-891503203', 'score': 0.7957592256503327, 'per_field': {'username': {'score': 0.903448275862069, 'weight': 0.6}, 'name': {'score': 0.6880701754385965, 'weight': 0.6}}}
```

Examples (after):

```
{'profileA_index': 6, 'profileA_id': '1ahmedDaouDi', 'profileB_index': 1396, 'profileB_id': 'ahmed-daoudi-aa693a230', 'score': 0.7638888888888888, 'per_field': {'username': {'score': 0.8055555555555555, 'weight': 0.6}, 'name': {'score': 0.7222222222222221, 'weight': 0.6}}}
```
```
{'profileA_index': 34, 'profileA_id': 'abdelhakmahm', 'profileB_index': 1260, 'profileB_id': 'abdelhak-madda-47638672', 'score': 0.79417854098161, 'per_field': {'username': {'score': 0.8536231884057971, 'weight': 0.6}, 'name': {'score': 0.7347338935574229, 'weight': 0.6}}}
```
```
{'profileA_index': 37, 'profileA_id': 'abdelilah_dourh', 'profileB_index': 742, 'profileB_id': 'abdelilahdahamou', 'score': 0.8192563563887092, 'per_field': {'username': {'score': 0.9127564102564102, 'weight': 0.6}, 'name': {'score': 0.7257563025210083, 'weight': 0.6}}}
```
```
{'profileA_index': 39, 'profileA_id': 'AbdellahSyani', 'profileB_index': 4194, 'profileB_id': 'abdellah-sbai-bb8b7616b', 'score': 0.7824414715719064, 'per_field': {'username': {'score': 0.8648829431438128, 'weight': 0.6}, 'name': {'score': 0.7, 'weight': 0.6}}}
```
```
{'profileA_index': 41, 'profileA_id': 'abdelmounaimhm2', 'profileB_index': 1703, 'profileB_id': 'abdelmounaim-hmamed-891503203', 'score': 0.7957592256503327, 'per_field': {'username': {'score': 0.903448275862069, 'weight': 0.6}, 'name': {'score': 0.6880701754385965, 'weight': 0.6}}}
```

---


## Mutual examples (sample)

### github->linkedin
- examples before: [('sabri-abdelaaziz', 'sabriabdelaaziz'), ('redaDaalabi2', 'reda-daalabi-1535ab1bb'), ('bousettayounes', 'bousettayounes'), ('abdarrhmanessetaoui', 'abderrhman-settaoui-33569b305'), ('RachidToumzine', 'rachid-toumzine')]
- examples after : [('sabri-abdelaaziz', 'sabriabdelaaziz'), ('redaDaalabi2', 'reda-daalabi-1535ab1bb'), ('bousettayounes', 'bousettayounes'), ('abdarrhmanessetaoui', 'abderrhman-settaoui-33569b305'), ('RachidToumzine', 'rachid-toumzine')]

### github->twitter
- examples before: [('ibouroum', 'IBouroummana'), ('elkhiari', 'Elkhiarii'), ('marouaneaddou', 'marouane'), ('hamza-douaioui', 'hamzadouri'), ('khalidxdev', 'khalid_baddou')]
- examples after : [('ibouroum', 'IBouroummana'), ('elkhiari', 'Elkhiarii'), ('marouaneaddou', 'marouane'), ('hamza-douaioui', 'hamzadouri'), ('khalidxdev', 'khalid_baddou')]

### linkedin->github
- examples before: [('younesse-elkars', 'YounesseElkars'), ('bousettayounes', 'bousettayounes'), ('nabil-cambiaso-533b95193', 'nabilcambiaso'), ('abdellatif-hassani', 'abdellatif-hassani'), ('ismail-harik-241b371b9', 'Ismailharik')]
- examples after : [('younesse-elkars', 'YounesseElkars'), ('ikram-choukhantri', 'ikramchoukhantri'), ('bousettayounes', 'bousettayounes'), ('nabil-cambiaso-533b95193', 'nabilcambiaso'), ('abdellatif-hassani', 'abdellatif-hassani')]

### linkedin->twitter
- examples before: [('ali-zaynoune', 'alizaynoune'), ('mehditouil', 'mehdi_jil'), ('ismail-ben-alla-bai', 'ismail_ben_alla'), ('imadeddarraz', 'ImadEddarraz'), ('kaoutharelbakouri', 'kawtarelbakouri')]
- examples after : [('ali-zaynoune', 'alizaynoune'), ('anouar-farroug', 'Anouar_Harry'), ('mehditouil', 'mehdi_jil'), ('ismail-ben-alla-bai', 'ismail_ben_alla'), ('imadeddarraz', 'ImadEddarraz')]

### twitter->github
- examples before: [('yassinelahlou', 'yassinelakchouch'), ('hamzajgameri', 'HamzaJemragi'), ('saraelkarii', 'saraelfar'), ('kaoutar_laouaj', 'Kaoutarlaouaj'), ('junik1337', 'junik1337')]
- examples after : [('yassinelahlou', 'yassinelakchouch'), ('medwf95', 'medwf'), ('hamzajgameri', 'HamzaJemragi'), ('saraelkarii', 'saraelfar'), ('kaoutar_laouaj', 'Kaoutarlaouaj')]

### twitter->linkedin
- examples before: [('khadijanacer', 'khadija-nacerdine-859156251'), ('ADez3_', 'adez3'), ('salah_jettioui', 'salaheddine-el-jettioui'), ('safoinme', 'safoinme'), ('ahmed_kamel_ir', 'ahmed-jamali')]
- examples after : [('khadijanacer', 'khadija-nacerdine-859156251'), ('RARHOUDANE', 'rachid-ait-rhoudane'), ('ADez3_', 'adez3'), ('salah_jettioui', 'salaheddine-el-jettioui'), ('safoinme', 'safoinme')]

