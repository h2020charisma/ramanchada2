# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/COVERAGE-REPORT/htmlcov/index.html)

| Name                                                                       |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/ramanchada2/\_\_init\_\_.py                                            |       29 |        4 |     86% |   133-136 |
| src/ramanchada2/\_\_main\_\_.py                                            |        0 |        0 |    100% |           |
| src/ramanchada2/auxiliary/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/ramanchada2/auxiliary/spectra/\_\_init\_\_.py                          |        1 |        0 |    100% |           |
| src/ramanchada2/auxiliary/spectra/datasets2/\_\_init\_\_.py                |       19 |        2 |     89% |   213-214 |
| src/ramanchada2/auxiliary/spectra/simulated/\_\_init\_\_.py                |        5 |        0 |    100% |           |
| src/ramanchada2/io/HSDS.py                                                 |       89 |       58 |     35% |20-64, 83-85, 95, 103-106, 110-122 |
| src/ramanchada2/io/\_\_init\_\_.py                                         |        0 |        0 |    100% |           |
| src/ramanchada2/io/experimental/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| src/ramanchada2/io/experimental/bw\_format.py                              |       16 |        0 |    100% |           |
| src/ramanchada2/io/experimental/neegala\_format.py                         |       12 |        1 |     92% |        12 |
| src/ramanchada2/io/experimental/rc1\_parser/\_\_init\_\_.py                |        1 |        0 |    100% |           |
| src/ramanchada2/io/experimental/rc1\_parser/binary\_readers.py             |      143 |      136 |      5% |16-37, 41-116, 120-125, 130-134, 138-202 |
| src/ramanchada2/io/experimental/rc1\_parser/io.py                          |       48 |       15 |     69% |18, 20, 22, 25-32, 60-64 |
| src/ramanchada2/io/experimental/rc1\_parser/third\_party\_readers.py       |       35 |        8 |     77% |     26-33 |
| src/ramanchada2/io/experimental/rc1\_parser/txt\_format\_readers.py        |      122 |      110 |     10% |11-43, 48-117, 121-141, 145-151, 155-166, 170-171 |
| src/ramanchada2/io/experimental/read\_csv.py                               |       11 |        7 |     36% |     10-19 |
| src/ramanchada2/io/experimental/read\_txt.py                               |       36 |        0 |    100% |           |
| src/ramanchada2/io/experimental/rruf\_format.py                            |       16 |        0 |    100% |           |
| src/ramanchada2/io/output/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/ramanchada2/io/output/write\_csv.py                                    |        3 |        1 |     67% |         5 |
| src/ramanchada2/io/simulated/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| src/ramanchada2/io/simulated/crystal/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| src/ramanchada2/io/simulated/crystal/discrete\_lines\_dat.py               |        6 |        0 |    100% |           |
| src/ramanchada2/io/simulated/crystal/discrete\_lines\_out.py               |       46 |        2 |     96% |    29, 34 |
| src/ramanchada2/io/simulated/lines\_from\_raw\_dat.py                      |        8 |        3 |     62% |      9-11 |
| src/ramanchada2/io/simulated/read\_simulated\_lines.py                     |       38 |       29 |     24% |     21-54 |
| src/ramanchada2/io/simulated/vasp/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/ramanchada2/io/simulated/vasp/vasp\_simulation\_dat.py                 |       19 |        0 |    100% |           |
| src/ramanchada2/misc/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| src/ramanchada2/misc/base\_class.py                                        |       20 |       11 |     45% |3, 6, 12, 16, 20-26 |
| src/ramanchada2/misc/constants.py                                          |       19 |        0 |    100% |           |
| src/ramanchada2/misc/exceptions.py                                         |        6 |        0 |    100% |           |
| src/ramanchada2/misc/plottable.py                                          |       14 |        1 |     93% |        17 |
| src/ramanchada2/misc/spectrum\_deco/\_\_init\_\_.py                        |        5 |        0 |    100% |           |
| src/ramanchada2/misc/spectrum\_deco/dynamically\_added.py                  |        3 |        0 |    100% |           |
| src/ramanchada2/misc/spectrum\_deco/spectrum\_constructor.py               |       21 |        1 |     95% |        23 |
| src/ramanchada2/misc/spectrum\_deco/spectrum\_filter.py                    |       20 |        1 |     95% |        23 |
| src/ramanchada2/misc/spectrum\_deco/spectrum\_method.py                    |       11 |        1 |     91% |        13 |
| src/ramanchada2/misc/types/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| src/ramanchada2/misc/types/fit\_peaks\_result.py                           |      104 |       31 |     70% |16-19, 44, 54, 64, 68, 100-108, 111, 133, 136-148 |
| src/ramanchada2/misc/types/peak\_candidates.py                             |       90 |       28 |     69% |27, 34, 37, 47-48, 51-56, 64, 68, 72, 76, 80, 88, 94, 101, 104-105, 108, 115, 121-122, 125, 131, 134 |
| src/ramanchada2/misc/types/positive\_not\_multiple.py                      |        2 |        0 |    100% |           |
| src/ramanchada2/misc/types/pydantic\_base\_model.py                        |       12 |        2 |     83% |    13, 23 |
| src/ramanchada2/misc/types/spectrum/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| src/ramanchada2/misc/types/spectrum/applied\_processings.py                |       50 |        2 |     96% |    29, 65 |
| src/ramanchada2/misc/types/spectrum/metadata.py                            |       51 |       10 |     80% |31-36, 46, 50, 58, 79 |
| src/ramanchada2/misc/utils/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| src/ramanchada2/misc/utils/argmin2d.py                                     |      105 |       23 |     78% |27-28, 44, 59-60, 68-83, 93, 103 |
| src/ramanchada2/misc/utils/ramanshift\_to\_wavelength.py                   |       17 |        2 |     88% |    17, 27 |
| src/ramanchada2/misc/utils/svd.py                                          |       13 |       10 |     23% |8-13, 17-20 |
| src/ramanchada2/protocols/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/ramanchada2/protocols/calibration.py                                   |      382 |       91 |     76% |25, 51, 65, 73-77, 82, 85, 88-97, 100, 104-106, 109-110, 164, 168-171, 175, 199, 222-232, 250-260, 268-269, 302, 305, 337, 391, 403, 447-448, 452, 501, 504-523, 538, 560-561, 625-626, 633-634, 653-684, 698, 700, 754, 756, 758, 769-770, 819, 840, 842, 844, 883-894, 897 |
| src/ramanchada2/spectral\_components/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| src/ramanchada2/spectral\_components/baseline/\_\_init\_\_.py              |        0 |        0 |    100% |           |
| src/ramanchada2/spectral\_components/baseline/analytical.py                |        0 |        0 |    100% |           |
| src/ramanchada2/spectral\_components/baseline/baseline\_base.py            |        3 |        3 |      0% |       1-5 |
| src/ramanchada2/spectral\_components/baseline/numerical.py                 |        3 |        3 |      0% |       1-5 |
| src/ramanchada2/spectral\_components/peak\_profile/\_\_init\_\_.py         |        2 |        0 |    100% |           |
| src/ramanchada2/spectral\_components/peak\_profile/delta.py                |       24 |       24 |      0% |      3-34 |
| src/ramanchada2/spectral\_components/peak\_profile/gauss.py                |       24 |       11 |     54% |13-18, 21-22, 26, 30, 34 |
| src/ramanchada2/spectral\_components/peak\_profile/voigt.py                |        4 |        1 |     75% |         8 |
| src/ramanchada2/spectral\_components/spectral\_component.py                |        8 |        3 |     62% |     11-13 |
| src/ramanchada2/spectral\_components/spectral\_component\_collection.py    |       49 |       34 |     31% |14-18, 21, 26-27, 30-31, 34, 38-39, 42-43, 46-65 |
| src/ramanchada2/spectral\_components/spectral\_peak.py                     |       27 |       13 |     52% | 14, 17-30 |
| src/ramanchada2/spectrum/\_\_init\_\_.py                                   |       16 |        0 |    100% |           |
| src/ramanchada2/spectrum/arithmetics/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/arithmetics/add.py                                |       20 |       11 |     45% |     18-29 |
| src/ramanchada2/spectrum/arithmetics/mul.py                                |       20 |       11 |     45% |     18-29 |
| src/ramanchada2/spectrum/arithmetics/sub.py                                |       20 |       11 |     45% |     18-29 |
| src/ramanchada2/spectrum/arithmetics/truediv.py                            |       20 |       11 |     45% |     17-28 |
| src/ramanchada2/spectrum/baseline/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/baseline/add\_baseline.py                         |       34 |        1 |     97% |        66 |
| src/ramanchada2/spectrum/baseline/baseline\_rc1.py                         |       45 |       12 |     73% | 19-29, 59 |
| src/ramanchada2/spectrum/baseline/moving\_minimum.py                       |       17 |        1 |     94% |        29 |
| src/ramanchada2/spectrum/calc/\_\_init\_\_.py                              |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/calc/central\_moments.py                          |       22 |       15 |     32% |     14-28 |
| src/ramanchada2/spectrum/calibration/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/calibration/by\_deltas.py                         |      133 |      109 |     18% |18-34, 37-46, 72-120, 132-150, 163-187, 211-231 |
| src/ramanchada2/spectrum/calibration/change\_x\_units.py                   |       21 |        3 |     86% |26, 42, 59 |
| src/ramanchada2/spectrum/calibration/normalize.py                          |       32 |        0 |    100% |           |
| src/ramanchada2/spectrum/calibration/scale\_xaxis.py                       |       18 |        3 |     83% |     31-33 |
| src/ramanchada2/spectrum/calibration/scale\_yaxis.py                       |        7 |        1 |     86% |        26 |
| src/ramanchada2/spectrum/calibration/set\_new\_xaxis.py                    |       10 |        3 |     70% |     27-29 |
| src/ramanchada2/spectrum/creators/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/creators/from\_cache\_or\_calc.py                 |       42 |        2 |     95% |    43, 62 |
| src/ramanchada2/spectrum/creators/from\_chada.py                           |        9 |        0 |    100% |           |
| src/ramanchada2/spectrum/creators/from\_delta\_lines.py                    |       23 |        0 |    100% |           |
| src/ramanchada2/spectrum/creators/from\_local\_file.py                     |       48 |        5 |     90% |44, 49-50, 55, 73 |
| src/ramanchada2/spectrum/creators/from\_simulation.py                      |       25 |       14 |     44% |     45-61 |
| src/ramanchada2/spectrum/creators/from\_spectral\_component\_collection.py |       10 |        3 |     70% |     25-27 |
| src/ramanchada2/spectrum/creators/from\_test\_spe.py                       |       12 |        0 |    100% |           |
| src/ramanchada2/spectrum/creators/from\_theoretical\_lines.py              |       18 |        8 |     56% |     30-37 |
| src/ramanchada2/spectrum/creators/hdr\_from\_multi\_exposure.py            |       17 |        9 |     47% |     24-32 |
| src/ramanchada2/spectrum/filters/\_\_init\_\_.py                           |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/filters/add\_gaussian\_noise.py                   |       17 |        0 |    100% |           |
| src/ramanchada2/spectrum/filters/add\_gaussian\_noise\_drift.py            |       24 |        0 |    100% |           |
| src/ramanchada2/spectrum/filters/add\_poisson\_noise.py                    |       16 |        0 |    100% |           |
| src/ramanchada2/spectrum/filters/convolve.py                               |       21 |        2 |     90% |    47, 50 |
| src/ramanchada2/spectrum/filters/drop\_spikes.py                           |       32 |        6 |     81% |56-58, 105-107 |
| src/ramanchada2/spectrum/filters/moving\_average.py                        |       14 |        3 |     79% | 26-28, 47 |
| src/ramanchada2/spectrum/filters/moving\_median.py                         |       16 |        4 |     75% |12-16, 38, 60 |
| src/ramanchada2/spectrum/filters/pad\_zeros.py                             |       14 |        7 |     50% |     29-36 |
| src/ramanchada2/spectrum/filters/resampling.py                             |       65 |        6 |     91% |68, 78-79, 165, 172-173 |
| src/ramanchada2/spectrum/filters/sharpen\_lines.py                         |       49 |       33 |     33% |38-51, 73-88, 112-115 |
| src/ramanchada2/spectrum/filters/smoothing.py                              |       30 |       16 |     47% |     42-57 |
| src/ramanchada2/spectrum/filters/trim\_axes.py                             |       16 |        2 |     88% |     32-33 |
| src/ramanchada2/spectrum/peaks/\_\_init\_\_.py                             |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/peaks/find\_peaks.py                              |      118 |       35 |     70% |84-86, 110-135, 139-161 |
| src/ramanchada2/spectrum/peaks/find\_peaks\_BayesianGaussianMixture.py     |       14 |        6 |     57% |     42-50 |
| src/ramanchada2/spectrum/peaks/fit\_peaks.py                               |       91 |       17 |     81% |46-50, 53-56, 65-68, 121-122, 135-136 |
| src/ramanchada2/spectrum/peaks/get\_fitted\_peaks.py                       |       18 |        9 |     50% |     45-56 |
| src/ramanchada2/spectrum/spectrum.py                                       |      166 |       35 |     79% |36, 65, 71, 74-77, 83, 92-94, 157-163, 185, 189-193, 200, 204-208, 217-220, 238, 243-245 |
| src/ramanchada2/theoretical\_lines/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| src/ramanchada2/theoretical\_lines/model\_from\_lines.py                   |       57 |       57 |      0% |      1-84 |
| tests/conftest.py                                                          |       29 |        1 |     97% |        35 |
| tests/end\_to\_end/test\_from\_cache\_or\_calc.py                          |       32 |        0 |    100% |           |
| tests/end\_to\_end/test\_generate\_and\_fit.py                             |       17 |        0 |    100% |           |
| tests/hierarchy\_test.py                                                   |        7 |        0 |    100% |           |
| tests/io/experimental/test\_input.py                                       |       10 |        1 |     90% |        10 |
| tests/io/simulated/crystal/test\_discrete\_lines\_dat.py                   |       12 |        0 |    100% |           |
| tests/io/simulated/crystal/test\_discrete\_lines\_out.py                   |        5 |        0 |    100% |           |
| tests/io/simulated/vasp/test\_vasp\_simulation\_dat.py                     |        7 |        0 |    100% |           |
| tests/misc/test\_argmin2d\_align.py                                        |       12 |        0 |    100% |           |
| tests/peak/pearson4\_test.py                                               |       50 |        0 |    100% |           |
| tests/protocols/test\_calibrationmodel.py                                  |      116 |        8 |     93% |77-79, 102-108 |
| tests/spectrum/creators/test\_from\_local\_file.py                         |       14 |        0 |    100% |           |
| tests/spectrum/filters/test\_resample\_NUDFT.py                            |       22 |        0 |    100% |           |
| tests/spectrum/filters/test\_resample\_spline.py                           |       16 |        0 |    100% |           |
| tests/spectrum/test\_calibration.py                                        |       11 |        0 |    100% |           |
| tests/spectrum/test\_filters.py                                            |       16 |        0 |    100% |           |
| tests/spectrum/test\_metadata.py                                           |       31 |        0 |    100% |           |
| tests/spectrum/test\_normalize.py                                          |       20 |        0 |    100% |           |
| tests/spectrum/test\_random\_generator\_seeds.py                           |       90 |        0 |    100% |           |
|                                                                  **TOTAL** | **3761** | **1163** | **69%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/h2020charisma/ramanchada2/COVERAGE-REPORT/badge.svg)](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/COVERAGE-REPORT/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/h2020charisma/ramanchada2/COVERAGE-REPORT/endpoint.json)](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/COVERAGE-REPORT/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fh2020charisma%2Framanchada2%2FCOVERAGE-REPORT%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/COVERAGE-REPORT/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.