# Multilingual normalizer dataset: stats

Total rows: 52698

| lang | language | template train/val/test | llm train/val/test | total |
|---|---|---|---|---|
| en | English | 2256/121/123 | 214/13/12 | 2739 |
| ms | Malay | 2273/102/125 | 208/15/15 | 2738 |
| id | Indonesian | 2246/138/116 | 209/10/9 | 2728 |
| zh | Mandarin Chinese | 2253/120/127 | 206/19/13 | 2738 |
| ta | Tamil | 2233/123/144 | 208/9/11 | 2728 |
| ta-LK | Tamil (Sri Lanka) | 2230/124/146 | 219/9/10 | 2738 |
| si | Sinhala | 2292/127/81 | 206/10/9 | 2725 |
| tl | Filipino (Tagalog) | 2256/110/134 | 209/15/12 | 2736 |
| ar | Arabic | 2262/123/115 | 207/12/12 | 2731 |
| fr | French | 2248/137/115 | 205/9/14 | 2728 |
| es | Spanish | 2219/150/131 | 205/19/13 | 2737 |
| de | German | 2242/145/113 | 200/20/16 | 2736 |
| it | Italian | 2273/117/110 | 209/9/16 | 2734 |
| pt | Portuguese | 2207/145/148 | 201/9/17 | 2727 |
| nl | Dutch | 2221/150/129 | 212/13/9 | 2734 |
| pl | Polish | 2262/121/117 | 174/14/13 | 2701 |
| ms-en | Malay-English code-switching (Malaysia) | 1314/87/99 | 0/0/0 | 1500 |
| en-ms | English-Malay code-switching (Malaysia) | 1340/52/108 | 0/0/0 | 1500 |
| zh-en | Mandarin-English code-switching (Malaysia) | 1365/40/95 | 0/0/0 | 1500 |
| zh-ms | Mandarin-Malay code-switching (Malaysia) | 1312/74/114 | 0/0/0 | 1500 |
| ta-en | Tamil-English code-switching (Malaysia) | 1338/61/101 | 0/0/0 | 1500 |
| ta-ms | Tamil-Malay code-switching (Malaysia) | 1309/65/126 | 0/0/0 | 1500 |

Slot / category counts (a template row counts once per slot):

money 5844, date 5667, digits 4704, int 4065, phone 3910, id 3887, email 3788, percent 3700, url 3653, time 3624, range 3269, year 3127, unit 2709, ordinal 2183, decimal 2094, big 1785, money:ta 1028, money:zh 969, date:zh 848, time:zh 842, id:en 821, time_plain:ta 686, date:ta 661, money:ms 590, url:en 577, date:en 537, percent:ta 440, percent:zh 425, time:ms 382, money:en 365, date:ms 363, time:en 326, digits:zh 294, range:en 287, int_small:ta 286, email:en 274, digits:ta 240, big:ta 226, big:zh 219, percent:ms 201, percent:en 192, negative 192, plain 191, unit_data:ta 178, mixed 177, fraction 174, unit_data:en 157, int_small:zh 151, unit_data:ms 141, int_small:ms 135, range:zh 125, phone:zh 121, digits:ms 116, year:zh 115, phone:ta 112, year:ta 111, big:ms 102, digits:en 101, int:zh 96, ordinal:ta 93, unit_temp:zh 90, int_small:en 87, int:ta 85, int:ms 84, unit_temp:ta 83, big:en 80, phone:en 58, year:ms 52, ordinal:zh 52, unit_temp:en 51, phone:ms 44, int:en 41, year:en 36, unit_temp:ms 27, ordinal:ms 21, ordinal:en 21
