# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                       |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/ramanchada2/\_\_init\_\_.py                                            |       30 |        4 |     87% |   131-134 |
| src/ramanchada2/\_\_main\_\_.py                                            |        0 |        0 |    100% |           |
| src/ramanchada2/auxiliary/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/ramanchada2/auxiliary/spectra/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/ramanchada2/auxiliary/spectra/datasets2/\_\_init\_\_.py                |       19 |        9 |     53% |186-190, 194-195, 199-200, 204 |
| src/ramanchada2/io/HSDS.py                                                 |       89 |       72 |     19% |22-66, 77-87, 93-100, 105-108, 112-124 |
| src/ramanchada2/io/\_\_init\_\_.py                                         |        0 |        0 |    100% |           |
| src/ramanchada2/io/experimental/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| src/ramanchada2/io/experimental/bw\_format.py                              |       16 |       13 |     19% |      9-24 |
| src/ramanchada2/io/experimental/rc1\_parser/\_\_init\_\_.py                |        1 |        0 |    100% |           |
| src/ramanchada2/io/experimental/rc1\_parser/binary\_readers.py             |      143 |      136 |      5% |13-34, 38-113, 117-122, 127-131, 135-199 |
| src/ramanchada2/io/experimental/rc1\_parser/io.py                          |       48 |       40 |     17% |12-44, 49-65 |
| src/ramanchada2/io/experimental/rc1\_parser/third\_party\_readers.py       |       35 |       16 |     54% |9-22, 26-33 |
| src/ramanchada2/io/experimental/rc1\_parser/txt\_format\_readers.py        |      122 |      110 |     10% |11-43, 48-117, 121-141, 145-151, 155-166, 170-171 |
| src/ramanchada2/io/experimental/read\_csv.py                               |       11 |        7 |     36% |     10-19 |
| src/ramanchada2/io/experimental/read\_txt.py                               |       14 |        9 |     36% | 11, 21-29 |
| src/ramanchada2/io/experimental/two\_column\_spe.py                        |        6 |        2 |     67% |     24-25 |
| src/ramanchada2/io/output/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/ramanchada2/io/output/write\_csv.py                                    |        3 |        1 |     67% |         5 |
| src/ramanchada2/io/simulated/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| src/ramanchada2/io/simulated/crystal/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| src/ramanchada2/io/simulated/crystal/discrete\_lines\_dat.py               |        6 |        0 |    100% |           |
| src/ramanchada2/io/simulated/crystal/discrete\_lines\_out.py               |       46 |        3 |     93% |21, 31, 36 |
| src/ramanchada2/io/simulated/lines\_from\_raw\_dat.py                      |        8 |        3 |     62% |     11-13 |
| src/ramanchada2/io/simulated/read\_simulated\_lines.py                     |       38 |       29 |     24% |     23-56 |
| src/ramanchada2/io/simulated/vasp/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/ramanchada2/io/simulated/vasp/vasp\_simulation\_dat.py                 |       19 |        0 |    100% |           |
| src/ramanchada2/misc/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| src/ramanchada2/misc/base\_class.py                                        |       20 |       11 |     45% |3, 6, 12, 16, 20-26 |
| src/ramanchada2/misc/constants.py                                          |       17 |       17 |      0% |     1-541 |
| src/ramanchada2/misc/exceptions.py                                         |        6 |        0 |    100% |           |
| src/ramanchada2/misc/plottable.py                                          |       14 |        2 |     86% |    10, 17 |
| src/ramanchada2/misc/spectrum\_deco/\_\_init\_\_.py                        |        5 |        0 |    100% |           |
| src/ramanchada2/misc/spectrum\_deco/dynamically\_added.py                  |        3 |        0 |    100% |           |
| src/ramanchada2/misc/spectrum\_deco/spectrum\_constructor.py               |       21 |        2 |     90% |    19, 23 |
| src/ramanchada2/misc/spectrum\_deco/spectrum\_filter.py                    |       20 |        1 |     95% |        23 |
| src/ramanchada2/misc/spectrum\_deco/spectrum\_method.py                    |       11 |        1 |     91% |        13 |
| src/ramanchada2/misc/types/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| src/ramanchada2/misc/types/fit\_peaks\_result.py                           |      104 |       40 |     62% |16-19, 27, 31, 44, 54, 64, 68, 71, 75-80, 100-108, 111, 133, 136-148 |
| src/ramanchada2/misc/types/peak\_candidates.py                             |       90 |       31 |     66% |29, 36, 39, 49-50, 53-58, 66, 70, 74, 78, 82, 86, 90, 93, 96, 103, 106-107, 110, 117, 120, 123-124, 127, 133, 136 |
| src/ramanchada2/misc/types/positive\_not\_multiple.py                      |       18 |        5 |     72% |14-16, 20-21 |
| src/ramanchada2/misc/types/pydantic\_base\_model.py                        |        8 |        1 |     88% |        11 |
| src/ramanchada2/misc/types/spectrum/\_\_init\_\_.py                        |        5 |        0 |    100% |           |
| src/ramanchada2/misc/types/spectrum/applied\_processings.py                |       50 |       15 |     70% |17-18, 29, 43, 46, 56-62, 65, 68, 73 |
| src/ramanchada2/misc/types/spectrum/metadata.py                            |       50 |        9 |     82% |35-40, 50, 54, 62 |
| src/ramanchada2/misc/utils/\_\_init\_\_.py                                 |        3 |        0 |    100% |           |
| src/ramanchada2/misc/utils/argmin2d.py                                     |      105 |       89 |     15% |12-17, 21-22, 26-27, 40-60, 67-82, 87-146 |
| src/ramanchada2/misc/utils/ramanshift\_to\_wavelength.py                   |       17 |       11 |     35% |5-7, 11-13, 17, 21-23, 27 |
| src/ramanchada2/misc/utils/svd.py                                          |       13 |       10 |     23% |8-13, 17-20 |
| src/ramanchada2/protocols/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/ramanchada2/protocols/calibration.py                                   |      190 |      190 |      0% |     1-335 |
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
| src/ramanchada2/spectrum/arithmetics/add.py                                |       20 |       11 |     45% |     20-31 |
| src/ramanchada2/spectrum/arithmetics/mul.py                                |       20 |       11 |     45% |     19-30 |
| src/ramanchada2/spectrum/arithmetics/sub.py                                |       20 |       11 |     45% |     19-30 |
| src/ramanchada2/spectrum/arithmetics/truediv.py                            |       20 |       11 |     45% |     19-30 |
| src/ramanchada2/spectrum/baseline/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/baseline/add\_baseline.py                         |       34 |        4 |     88% |21-22, 36, 70 |
| src/ramanchada2/spectrum/baseline/baseline\_rc1.py                         |       45 |       26 |     42% |18-28, 33-48, 58, 68 |
| src/ramanchada2/spectrum/baseline/moving\_minimum.py                       |       17 |        5 |     71% |12-18, 35, 44 |
| src/ramanchada2/spectrum/calc/\_\_init\_\_.py                              |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/calc/central\_moments.py                          |       22 |       15 |     32% |     13-27 |
| src/ramanchada2/spectrum/calibration/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/calibration/by\_deltas.py                         |      133 |      109 |     18% |15-31, 34-43, 69-117, 129-147, 160-184, 208-228 |
| src/ramanchada2/spectrum/calibration/change\_x\_units.py                   |       21 |        4 |     81% |15, 22, 30, 38 |
| src/ramanchada2/spectrum/calibration/normalize.py                          |       24 |       16 |     33% |     28-43 |
| src/ramanchada2/spectrum/calibration/scale\_xaxis.py                       |       18 |        3 |     83% |     19-21 |
| src/ramanchada2/spectrum/calibration/scale\_yaxis.py                       |        7 |        1 |     86% |        14 |
| src/ramanchada2/spectrum/calibration/set\_new\_xaxis.py                    |       10 |        3 |     70% |     15-17 |
| src/ramanchada2/spectrum/creators/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/creators/from\_cache\_or\_calc.py                 |       41 |       32 |     22% |     17-55 |
| src/ramanchada2/spectrum/creators/from\_chada.py                           |        9 |        2 |     78% |     13-14 |
| src/ramanchada2/spectrum/creators/from\_delta\_lines.py                    |       23 |        1 |     96% |        47 |
| src/ramanchada2/spectrum/creators/from\_local\_file.py                     |       40 |       29 |     28% |     42-74 |
| src/ramanchada2/spectrum/creators/from\_simulation.py                      |       19 |        8 |     58% |     40-48 |
| src/ramanchada2/spectrum/creators/from\_spectral\_component\_collection.py |       10 |        3 |     70% |     22-24 |
| src/ramanchada2/spectrum/creators/from\_test\_spe.py                       |       13 |        6 |     54% |     25-31 |
| src/ramanchada2/spectrum/creators/from\_theoretical\_lines.py              |       18 |        8 |     56% |     33-40 |
| src/ramanchada2/spectrum/creators/hdr\_from\_multi\_exposure.py            |       17 |        9 |     47% |     24-32 |
| src/ramanchada2/spectrum/filters/\_\_init\_\_.py                           |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/filters/add\_gaussian\_noise.py                   |       17 |        3 |     82% | 32-33, 40 |
| src/ramanchada2/spectrum/filters/add\_gaussian\_noise\_drift.py            |       24 |       15 |     38% | 16-32, 71 |
| src/ramanchada2/spectrum/filters/add\_poisson\_noise.py                    |       17 |        3 |     82% | 32-33, 39 |
| src/ramanchada2/spectrum/filters/convolve.py                               |       21 |        2 |     90% |    44, 47 |
| src/ramanchada2/spectrum/filters/drop\_spikes.py                           |       36 |       16 |     56% |23-30, 68-75 |
| src/ramanchada2/spectrum/filters/moving\_average.py                        |       14 |        3 |     79% | 24-26, 34 |
| src/ramanchada2/spectrum/filters/moving\_median.py                         |       16 |        4 |     75% |12-16, 33, 42 |
| src/ramanchada2/spectrum/filters/pad\_zeros.py                             |       14 |        7 |     50% |     14-21 |
| src/ramanchada2/spectrum/filters/resampling.py                             |       27 |       15 |     44% | 20-33, 44 |
| src/ramanchada2/spectrum/filters/sharpen\_lines.py                         |       49 |       33 |     33% |22-35, 44-59, 68-71 |
| src/ramanchada2/spectrum/filters/smoothing.py                              |       30 |       18 |     40% |     26-43 |
| src/ramanchada2/spectrum/filters/trim\_axes.py                             |       16 |        8 |     50% |     19-26 |
| src/ramanchada2/spectrum/peaks/\_\_init\_\_.py                             |        3 |        0 |    100% |           |
| src/ramanchada2/spectrum/peaks/find\_peaks.py                              |      118 |       40 |     66% |23, 30, 52, 57-59, 83-108, 112-134, 175-176 |
| src/ramanchada2/spectrum/peaks/find\_peaks\_BayesianGaussianMixture.py     |       14 |        6 |     57% |     22-30 |
| src/ramanchada2/spectrum/peaks/fit\_peaks.py                               |       90 |       24 |     73% |48-52, 55-58, 66-76, 99-100, 113-114, 132-133 |
| src/ramanchada2/spectrum/peaks/get\_fitted\_peaks.py                       |       17 |        9 |     47% |     20-31 |
| src/ramanchada2/spectrum/spectrum.py                                       |      145 |       50 |     66% |33, 41-43, 59, 62, 65-68, 71, 74, 78, 83-85, 90-92, 96-98, 110-112, 124, 144, 147-153, 180, 185-188, 192, 196, 200-207, 211-213 |
| src/ramanchada2/theoretical\_lines/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| src/ramanchada2/theoretical\_lines/model\_from\_lines.py                   |       57 |       57 |      0% |      3-87 |
| tests/conftest.py                                                          |       22 |        0 |    100% |           |
| tests/end\_to\_end/test\_generate\_and\_fit.py                             |       17 |        0 |    100% |           |
| tests/hierarchy\_test.py                                                   |        7 |        0 |    100% |           |
| tests/io/experimental/test\_input.py                                       |       10 |        1 |     90% |        10 |
| tests/io/simulated/crystal/test\_discrete\_lines\_dat.py                   |       12 |        0 |    100% |           |
| tests/io/simulated/crystal/test\_discrete\_lines\_out.py                   |        5 |        0 |    100% |           |
| tests/io/simulated/vasp/test\_vasp\_simulation\_dat.py                     |        7 |        0 |    100% |           |
| tests/peak/pearson4\_test.py                                               |       50 |        0 |    100% |           |
| tests/spectrum/test\_calibration.py                                        |       11 |        0 |    100% |           |
| tests/spectrum/test\_filters.py                                            |       16 |        0 |    100% |           |
| tests/spectrum/test\_metadata.py                                           |       31 |        0 |    100% |           |
|                                                                  **TOTAL** | **3125** | **1623** | **48%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/h2020charisma/ramanchada2/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/h2020charisma/ramanchada2/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fh2020charisma%2Framanchada2%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/h2020charisma/ramanchada2/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.