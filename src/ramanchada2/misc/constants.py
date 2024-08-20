from .. import spectrum as rc2spectrum

from .utils.ramanshift_to_wavelength import (abs_nm_to_shift_cm_1_dict,
                                             shift_cm_1_to_abs_nm_dict)

PST_RS_dict = {620.9: 16, 795.8: 10, 1001.4: 100, 1031.8: 27, 1155.3: 13, 1450.5: 8,
               1583.1: 12, 1602.3: 28, 2852.4: 9, 2904.5: 13, 3054.3: 32}


neon_nist_wl_nm = {200.7009: 80,
                   202.55599999999998: 80,
                   208.54659999999998: 150,
                   209.61060000000003: 200,
                   209.6248: 120,
                   256.2123: 80,
                   256.7121: 90,
                   262.3107: 80,
                   262.98850000000004: 80,
                   263.6069: 90,
                   263.82890000000003: 80,
                   264.40970000000004: 80,
                   276.2921: 80,
                   279.20189999999997: 90,
                   279.4221: 80,
                   280.9485: 100,
                   290.6592: 80,
                   290.6816: 80,
                   291.0061: 90,
                   291.0408: 90,
                   291.11379999999997: 80,
                   291.5122: 80,
                   292.5618: 80,
                   293.2103: 80,
                   294.0653: 80,
                   294.6044: 90,
                   295.5725: 150,
                   296.3236: 150,
                   296.71840000000003: 150,
                   297.2997: 100,
                   297.47189: 30,
                   297.9461: 100,
                   298.26696000000004: 30,
                   300.1668: 150,
                   301.7311: 120,
                   302.7016: 300,
                   302.8864: 300,
                   303.07869999999997: 100,
                   303.4461: 120,
                   303.59229999999997: 100,
                   303.772: 100,
                   303.9586: 100,
                   304.40880000000004: 100,
                   304.5556: 100,
                   304.7556: 120,
                   305.43449999999996: 100,
                   305.46770000000004: 100,
                   305.73906999999997: 30,
                   305.91060000000004: 100,
                   306.2491: 100,
                   306.3301: 100,
                   307.0887: 100,
                   307.1529: 100,
                   307.5731: 100,
                   308.8166: 120,
                   309.2092: 100,
                   309.2901: 120,
                   309.4006: 100,
                   309.51030000000003: 100,
                   309.7131: 100,
                   311.798: 100,
                   311.816: 120,
                   314.1332: 300,
                   314.3721: 100,
                   314.8681: 100,
                   316.4429: 100,
                   316.5648: 100,
                   318.8743: 100,
                   319.4579: 120,
                   319.85859999999997: 500,
                   320.8965: 60,
                   320.9356: 120,
                   321.37350000000004: 120,
                   321.4329: 150,
                   321.8193: 150,
                   322.4818: 120,
                   322.9573: 120,
                   323.007: 200,
                   323.0419: 120,
                   323.2022: 120,
                   323.2372: 150,
                   324.3396: 100,
                   324.4095: 100,
                   324.8345: 100,
                   325.0355: 100,
                   329.7726: 150,
                   330.974: 150,
                   331.97220000000004: 300,
                   332.3745: 1000,
                   332.71529999999996: 150,
                   332.9158: 100,
                   333.48359999999997: 200,
                   334.4395: 150,
                   334.5453: 300,
                   334.5829: 150,
                   335.5016: 200,
                   335.78200000000004: 120,
                   336.0597: 200,
                   336.2161: 120,
                   336.2707: 100,
                   336.7218: 120,
                   336.98076000000003: 50,
                   336.99072: 70,
                   337.1799: 100,
                   337.8216: 500,
                   338.8417: 150,
                   338.8945: 120,
                   339.27979999999997: 300,
                   340.48220000000003: 100,
                   340.6947: 120,
                   341.3148: 100,
                   341.69140000000004: 120,
                   341.7688: 120,
                   341.79031: 50,
                   341.80055000000004: 5,
                   342.8687: 120,
                   344.77024: 20,
                   345.41944: 10,
                   345.661: 100,
                   345.9321: 100,
                   346.05237: 10,
                   346.43382: 10,
                   346.65781000000004: 20,
                   347.25706: 50,
                   347.95189999999997: 150,
                   348.0718: 200,
                   348.1933: 200,
                   349.80636: 10,
                   350.12159: 20,
                   351.51902: 20,
                   352.04711000000003: 100,
                   354.28470000000004: 120,
                   355.78049999999996: 120,
                   356.1198: 100,
                   356.8502: 250,
                   357.4181: 100,
                   357.4612: 200,
                   359.35257: 50,
                   359.36388999999997: 30,
                   360.01685000000003: 10,
                   363.3664: 10,
                   364.3927: 150,
                   366.40729999999996: 200,
                   368.22420999999997: 10,
                   368.57352000000003: 10,
                   369.42130000000003: 200,
                   370.12244: 4,
                   370.9622: 150,
                   371.3079: 250,
                   372.7107: 250,
                   376.6259: 800,
                   377.7133: 1000,
                   381.84270000000004: 100,
                   382.9749: 120,
                   421.9745: 150,
                   423.38500000000005: 100,
                   425.0649: 120,
                   436.9862: 120,
                   437.93999999999994: 70,
                   437.95500000000004: 150,
                   438.5059: 100,
                   439.1991: 200,
                   439.799: 150,
                   440.9299: 150,
                   441.3215: 100,
                   442.13890000000004: 100,
                   442.85159999999996: 100,
                   442.8634: 100,
                   443.09040000000005: 150,
                   443.0942: 150,
                   445.7049: 120,
                   452.27200000000005: 100,
                   453.77545: 100,
                   456.90569999999997: 100,
                   470.43949000000003: 150,
                   470.88594: 120,
                   471.00649999999996: 100,
                   471.20633: 150,
                   471.5344: 150,
                   475.2732: 50,
                   478.89258: 100,
                   479.02195: 50,
                   482.7338: 100,
                   488.49170000000004: 100,
                   500.51587: 50,
                   503.77511999999996: 50,
                   514.49384: 50,
                   533.07775: 60,
                   534.10938: 100,
                   534.32834: 60,
                   540.05618: 200,
                   556.27662: 50,
                   565.66588: 50,
                   571.92248: 50,
                   574.82985: 50,
                   576.4418800000001: 70,
                   580.44496: 50,
                   582.01558: 50,
                   585.24879: 200,
                   587.2827500000001: 50,
                   588.18952: 100,
                   590.24623: 5,
                   590.64294: 5,
                   594.48342: 50,
                   596.5471: 50,
                   597.46273: 50,
                   597.5534: 60,
                   598.79074: 15,
                   602.99969: 100,
                   607.43377: 100,
                   609.61631: 30,
                   612.8449899999999: 10,
                   614.3062600000001: 100,
                   616.35939: 100,
                   618.2146: 15,
                   621.72812: 100,
                   626.6495: 100,
                   630.47889: 10,
                   632.81646: 30,
                   633.4427800000001: 100,
                   638.29917: 100,
                   640.2248: 200,
                   650.65281: 150,
                   653.28822: 10,
                   659.89529: 100,
                   665.2092700000001: 15,
                   667.82762: 50,
                   671.7043: 7,
                   692.94673: 1000,
                   702.40504: 300,
                   703.24131: 800,
                   705.12923: 20,
                   705.91074: 100,
                   717.39381: 800,
                   721.3199999999999: 150,
                   723.5188: 150,
                   724.51666: 800,
                   734.3945: 150,
                   747.24386: 30,
                   748.88712: 300,
                   749.2102: 100,
                   752.2818: 150,
                   753.57741: 300,
                   754.4044299999999: 130,
                   772.4623300000001: 1,
                   774.0738: 120,
                   783.9052899999999: 2,
                   792.6201: 120,
                   792.71177: 3,
                   793.69961: 13,
                   794.3181400000001: 80,
                   808.2457999999999: 60,
                   808.4345000000001: 100,
                   811.85492: 40,
                   812.89108: 12,
                   813.64054: 170,
                   825.9379000000001: 30,
                   826.4807000000001: 100,
                   826.60772: 70,
                   826.71162: 10,
                   830.03258: 300,
                   831.4995000000001: 100,
                   836.57466: 50,
                   837.2106: 100,
                   837.7608: 800,
                   841.71606: 30,
                   841.84274: 250,
                   846.33575: 40,
                   848.44435: 13,
                   849.53598: 700,
                   854.46958: 15,
                   857.13524: 30,
                   859.12584: 400,
                   863.4647000000001: 350,
                   864.70411: 60,
                   865.4383099999999: 600,
                   865.5522000000001: 80,
                   866.8255999999999: 100,
                   867.94925: 130,
                   868.19211: 150,
                   870.41116: 30,
                   877.1656300000001: 100,
                   878.06226: 600,
                   878.3753300000001: 400,
                   883.0907199999999: 6,
                   885.38668: 300,
                   886.53063: 20,
                   886.57552: 150,
                   891.9500599999999: 60,
                   898.85564: 20,
                   907.9462: 100,
                   914.86716: 120,
                   920.1759099999999: 90,
                   922.0060100000001: 60,
                   922.1580099999999: 20,
                   922.66903: 20,
                   927.55196: 9,
                   928.7563: 200,
                   930.0852699999999: 80,
                   931.0583899999999: 8,
                   931.39726: 30,
                   932.65068: 70,
                   937.33078: 15,
                   942.5378800000001: 50,
                   945.9209500000001: 30,
                   948.66818: 50,
                   953.4162899999999: 60,
                   954.7404899999999: 30,
                   957.7013000000001: 120,
                   966.54197: 180,
                   980.8860000000001: 100,
                   1029.5417400000001: 4,
                   1056.24075: 80,
                   1079.80429: 60,
                   1084.44772: 90,
                   1114.3020000000001: 300,
                   1117.7523999999999: 500,
                   1139.04339: 150,
                   1140.91343: 90,
                   1152.27459: 300,
                   1152.5019399999999: 150,
                   1153.63445: 90,
                   1160.15366: 30,
                   1161.40807: 130,
                   1168.80017: 30,
                   1176.67924: 150,
                   1178.90435: 130,
                   1178.98891: 30,
                   1198.4912: 70,
                   1206.6334000000002: 200,
                   1245.9388999999999: 40,
                   1268.9200999999998: 60,
                   1291.2014: 80,
                   1321.9241: 40,
                   1523.0714: 50,
                   1716.1929: 20,
                   1803.5812: 20,
                   1808.3181: 40,
                   1808.3263: 9,
                   1822.1087: 15,
                   1822.7015999999999: 13,
                   1827.6642: 140,
                   1828.2614: 100,
                   1830.3967: 70,
                   1835.9094: 20,
                   1838.4826: 60,
                   1838.9937000000002: 90,
                   1840.2836: 40,
                   1842.2401999999997: 60,
                   1845.864: 13,
                   1847.58: 40,
                   1859.1541000000002: 70,
                   1859.7698: 100,
                   1861.8908: 16,
                   1862.5158999999999: 20,
                   2104.127: 30,
                   2170.811: 30,
                   2224.736: 13,
                   2242.814: 13,
                   2253.038: 80,
                   2266.179: 13,
                   2310.048: 25,
                   2326.027: 40,
                   2337.296: 50,
                   2356.5330000000004: 30,
                   2363.648: 170,
                   2370.166: 12,
                   2370.913: 60,
                   2395.1400000000003: 110,
                   2395.643: 50,
                   2397.816: 60,
                   2409.857: 11,
                   2416.143: 20,
                   2424.9610000000002: 30,
                   2436.5009999999997: 70,
                   2437.161: 40,
                   2444.786: 20,
                   2445.939: 30,
                   2477.6490000000003: 17,
                   2492.889: 30,
                   2516.17: 13,
                   2552.433: 50,
                   2838.62: 6,
                   3020.049: 6,
                   3317.3089999999997: 8,
                   3335.238: 17,
                   3389.9809999999998: 5,
                   3390.3019999999997: 4,
                   3391.31: 12,
                   3413.1339999999996: 4,
                   3447.143: 6,
                   3583.4809999999998: 8}


polystyrene_rs = {620.9: 16, 795.8: 10, 1001.4: 100, 1031.8: 27, 1155.3: 13, 1450.5: 8,
                  1583.1: 12, 1602.3: 28, 2852.4: 9, 2904.5: 13, 3054.3: 32}


neon_rs_dict = {
    785: {k: v for k, v in abs_nm_to_shift_cm_1_dict(neon_nist_wl_nm, 785).items() if 100 < k < 3000},
    633: {k: v for k, v in abs_nm_to_shift_cm_1_dict(neon_nist_wl_nm, 633).items() if 100 < k < 4500},
    532: {k: v for k, v in abs_nm_to_shift_cm_1_dict(neon_nist_wl_nm, 532).items() if 100 < k < 4000},
}

neon_wl_dict = {
    785: {k: v for k, v in shift_cm_1_to_abs_nm_dict(neon_rs_dict[785], 785).items()},
    633: {k: v for k, v in shift_cm_1_to_abs_nm_dict(neon_rs_dict[633], 633).items()},
    532: {k: v for k, v in shift_cm_1_to_abs_nm_dict(neon_rs_dict[532], 532).items()},
}


neon_rs_spe = {
    785: rc2spectrum.from_delta_lines(neon_rs_dict[785], xcal=lambda x: x, nbins=4500).convolve('gaussian', sigma=1),
    633: rc2spectrum.from_delta_lines(neon_rs_dict[633], xcal=lambda x: x, nbins=4500).convolve('gaussian', sigma=2),
    532: rc2spectrum.from_delta_lines(neon_rs_dict[532], xcal=lambda x: x, nbins=4500).convolve('gaussian', sigma=3),
}

neon_wl_spe = {
    785: rc2spectrum.from_delta_lines(neon_wl_dict[785], xcal=lambda x: x/22.85 + 788, nbins=4500
                                      ).convolve('gaussian', sigma=1.5),
    633: rc2spectrum.from_delta_lines(neon_wl_dict[633], xcal=lambda x: x/17.64 + 633, nbins=4500
                                      ).convolve('gaussian', sigma=1.5),
    532: rc2spectrum.from_delta_lines(neon_wl_dict[532], xcal=lambda x: x/31.69 + 535, nbins=4500
                                      ).convolve('gaussian', sigma=1.5),
}


neon_wl_785_nist_dict = {
    792.6841221569385: 5.37437266023976, 793.7330415754924: 0.5689277951354417,
    794.3457330415755: 3.501094091821991, 808.4285595693893: 7.202537154952434,
    811.8949671772428: 1.7505470455873593, 812.945295404814: 0.5251641209139684,
    813.6892778993436: 7.439824945294173, 825.9868433097755: 1.3120972814694163,
    826.5665828089599: 7.94695465954125, 830.0568927789934: 13.129102844632756,
    831.5448577680525: 4.3763676148399115, 836.6214442013129: 2.1881838071971296,
    837.234135667396: 4.376367614845547, 837.8030634573304: 35.010940919002735,
    841.8664541909187: 12.03028296382178, 846.3807439824946: 1.750547045586516,
    848.4814004376367: 0.568927789934395, 849.5754923413567: 30.634573304125958,
    854.5207877461706: 0.6564551422317848, 857.1903719912473: 1.3129102839767155,
    859.1597374179431: 17.50547045950853, 863.4923413566739: 15.317286652068542,
    865.469184130712: 29.20051773802159, 866.8621444201312: 4.376367614840063,
    868.0: 5.689277899344474, 868.218818380744: 6.564551422319682,
    870.4507658643327: 1.3129102839772355, 877.190371991244: 4.376367613743546,
    878.1094091903724: 26.2582056879645, 878.4157549234137: 17.505470458778227,
    883.1422319474835: 0.262582056892803, 885.4179431072209: 13.129102844635762,
    886.5947874281137: 7.437997551100201, 891.9824945295405: 2.6258205687594547,
    898.8971553610503: 0.8752735229757134, 908.0: 4.376367614838839,
    914.9146608315099: 5.251641137834712, 920.2100656455142: 3.9387308533338343,
    922.0688951698448: 3.3797527091026245, 927.6061269146609: 0.39387308533920273,
    928.7877461706784: 8.752735229757006, 930.144420131291: 3.5010940918132034,
    931.1072210065647: 0.35010940916879774, 931.4573304157549: 1.3129102844605525,
    932.6827133479212: 3.0634573302989097, 937.3654266958424: 0.6564551422320757,
    942.5733041575493: 2.188183807192464, 945.9431072210066: 1.3129102839781117,
    948.7002188183808: 2.188183807191779, 953.4704595185996: 2.6258205687594476,
    954.7833698030635: 1.3129102839777373, 957.7592997811817: 5.2516411378370345,
    966.5995623632385: 7.877461706780519, 980.910284463895: 4.376367614839546
}

neon_wl_633_nist_dict = {
    638.3287981859411: 5.6689342403499765, 640.2562358276643: 11.337868480727586,
    650.6870748299319: 8.503401360545043, 653.3514739229025: 0.566893424036351,
    659.9274376417234: 5.668934240348053, 665.2562358276643: 0.8503401360546983,
    667.8639455782313: 2.8344671200402334, 671.7755102040816: 0.39682539682552465,
    692.9773242630386: 56.68934240353873, 702.4444444444445: 17.006802721074873,
    703.2947845804988: 45.351473922858105, 705.1655328798186: 1.1337868476493322,
    705.9591836734694: 5.668934240348062, 717.4671201814059: 45.351473922836774,
    721.3786848072563: 8.503401360545148, 723.5895691609977: 8.50340136054297,
    724.5532879818594: 45.351473922833904, 734.4739229024943: 8.503401360542986,
    747.2857142857143: 1.7006802717293952, 748.9297052154195: 17.00680272108454,
    749.2698412698412: 5.6689342403646865, 752.3310657596371: 8.503401360527796,
    753.6349206349206: 17.006802721048707, 754.4852607709751: 7.369614512457517,
    772.5124716553288: 0.05668934240365808, 774.156462585034: 6.802721088429488,
    783.9637188208617: 0.11337868480724644, 792.6951245635115: 6.971961625854252,
    793.77097505694: 0.7369614305162637, 794.3945578231699: 4.535147381418968,
    808.4417565814348: 9.329816961551543, 811.9115646254535: 2.2675736482945825,
    812.9319727898597: 0.6802720427342306, 813.6689342404153: 9.63718817885658,
    825.9704271162868: 1.6978631010464373, 826.5881899888147: 10.231887665802304,
    830.1088435374149: 17.00680272107302, 831.5827664399093: 5.668934240348589,
    836.6281179138335: 2.8344671202589966, 837.2517006802728: 5.6689342195710095,
    837.8185941043079: 45.351473923298364, 841.8913979246645: 15.8101670633123,
    846.3786848072561: 2.26757369591529, 848.4761904759695: 0.7369614416713715,
    849.6099773242671: 39.68253967290885, 854.5419501133787: 0.8503401360543106,
    857.2063492063492: 1.7006802717279004, 859.1904761904761: 22.67573696142483,
    863.4988662131518: 19.841269841214043, 864.7460316744742: 3.401354571136033,
    865.4928040619619: 38.396369552201875, 866.9002267573604: 5.668934237269633,
    868.034013605442: 7.369614508413964, 868.26077097504: 8.503401360112813,
    870.4716553287982: 1.7006802717287801, 877.2176870748299: 5.668934240361571,
    878.1247165532881: 34.01360544219461, 878.4081632653063: 22.675736961434616,
    883.1700680272108: 0.3401360544216929
}

neon_wl_532_nist_dict = {
    540.0804670242978: 6.311139160610751, 556.3000946670874: 1.5777847897339283,
    565.7036920164089: 1.5777847897345527, 571.9517197854212: 1.5777847897346757,
    574.8548437993057: 1.5777847897345527, 576.4641842852635: 2.2088987059726164,
    580.4717576522562: 1.5777847897345527, 582.0495424424108: 1.5777847897345523,
    585.2682234143263: 6.311139160610822, 587.3193436415273: 1.5777847897341903,
    588.234458819817: 3.1555695802011297, 590.285579047018: 0.15777847901546926,
    590.6642473966551: 0.15777847901546949, 594.5140422846324: 1.5777847897344977,
    596.5651625118334: 1.5777847897347166, 597.5336867539808: 3.5612432617200778,
    598.837172609656: 0.4733354370464895, 603.0340801514673: 3.155569580200976,
    607.4518775639002: 3.155569580201294, 609.6607762701167: 0.9466708739111769,
    612.8794572420321: 0.31555695803092476, 614.3310192489744: 3.1555695802009724,
    616.3821394761754: 3.1555695802010364, 618.2439255285578: 0.4733354370463865,
    621.7466077627012: 3.1555695802013357, 626.6692963079836: 3.1555695802012638,
    630.5190911959609: 0.31555695803092476, 632.8542126853799: 0.9466708729394551,
    633.4853266014544: 3.1555695792292964, 638.3449037551278: 3.1555695802011074,
    640.2698011991164: 6.31113916061083, 650.6831808141369: 4.733354370433662,
    653.3338592615967: 0.3155569580309239, 659.928999684443: 3.155569580201134,
    665.2303565793626: 0.4733354370463867, 667.8494793310192: 1.5777847897339252,
    671.7308299147995: 0.22088987062164733
}


neon_rs_532_nist_dict = abs_nm_to_shift_cm_1_dict(neon_wl_532_nist_dict, 532)
neon_rs_633_nist_dict = abs_nm_to_shift_cm_1_dict(neon_wl_633_nist_dict, 633)
neon_rs_785_nist_dict = abs_nm_to_shift_cm_1_dict(neon_wl_785_nist_dict, 785)

neon_wl_D3_3 = [
    533.07775, 540.05616, 556.27662, 565.66588, 571.92248, 574.82985, 576.44188,
    580.44496, 580.44496, 582.01558, 585.24878, 587.28275, 588.1895, 590.24623,
    594.4834, 596.5471, 598.79074, 602.99968, 607.43376, 609.6163, 612.84498,
    614.30627, 616.35937, 618.2146, 621.72812, 626.64952, 630.47893, 633.44276,
    638.29914, 640.2248, 650.65277, 653.28824, 659.89528, 667.82766, 671.7043,
    692.94672, 702.405, 703.24128, 705.91079, 717.3938, 724.51665, 748.88712,
    753.57739, 754.40439, 794.31805, 808.24576, 811.85495, 813.64061, 830.03248,
    836.57464, 837.7607, 846.33569, 849.53591, 854.46952, 857.13535, 859.12583,
    863.46472, 870.41122, 877.16575, 878.37539, 885.38669, 891.95007, 898.85564,
    898.85564, 914.8672, 920.17588, 927.55191, 930.08532, 932.65072, 937.33079,
    942.53797, 945.9211, 948.66825, 953.4164, 954.74052, 966.542,
]

neon_wl_D3_3_dict = dict(zip(neon_wl_D3_3, [1]*len(neon_wl_D3_3)))

NEON_WL = {
    785: neon_wl_785_nist_dict,
    633: neon_wl_633_nist_dict,
    532: neon_wl_532_nist_dict
}

