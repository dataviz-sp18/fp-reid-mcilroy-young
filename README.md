# RNN Occlusion visualization

## Setup

To run the code you will need `Python3` with the [`simple_id`](https://github.com/reidmcy/simple_id) package installed, along with `pytorch`. `simple_id` was created by me and contains all the NN handling and uses `pytorch` as its backend. The raw `Python` version of the code is in displayed in `demo.ipynb` or `demo.Rmd` if you prefer Rmarkdown. The demo webapp is hosted at [shiny.reidmcy.com/fp](http://shiny.reidmcy.com/fp) but reticulate causes it to crash (which should not even be possible in `R`) whenever anything is imported so you milage will vary.

In addition to the data files provide in this repo, you will need to download the word2vec embedding used by the RNN. These have been converted from a gensim style array into a zip file, which can be found at [reidmcy.com/assets/w2v.zip](http://reidmcy.com/assets/w2v.zip) (`1.3G`). this file makes look up a file operation instead of being entirely in memory as the zip file contains one file for each word in the vocab with the word as it's title and the vector as its value. There is no need to unzip the files, in fact the code requires it remain zipped, although allowing for unzipping could lead to significant run time improvements at the cost of massive disk and filesystem loads, a few well calibrated zip files is likely the optimal solution.

An example of the iamges created is shown below.

![occ](images/occ.png)
