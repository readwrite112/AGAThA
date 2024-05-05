# [PPoPP24] AGAThA: Fast and Efficient GPU Acceleration of Guided Sequence Alignment for Long Read Mapping [![DOI](https://zenodo.org/badge/725514536.svg)](https://zenodo.org/doi/10.5281/zenodo.10225634)

## !!! Important Notice !!! 
**This repository is currently undergoing a major update**.

It is strongly recommended to **revisit this repository after a new release is made**. 

## Getting Started

### 1. Environment Setup with Docker
```properties
cd docker
bash build.sh
bash launch.sh
```

### 2. Datasets & Building AGAThA
A sample dataset can be found in `dataset/`.
AGAThA can be built by running the following code:

```properties
cd AGAThA
bash build.sh
cd ..
```

## AGAThA Details

AGAThA was built on top of [GASAL2](https://github.com/nahmedraja/GASAL2).

### 1. AGAThA Options
Using ```AGAThA.sh```, we can use the following options for AGAThA.
```
-m  the match score
-x  the mismatch penatly
-q  the gap open penalty
-r  the gap extension penalty
-z  the termination threshold
-w  the band width in the score table
```
### 2. AGAThA Input
AGAThA requires two datasets as input; 
* a fasta file with reference sequences labeled with sequence indices
* a fasta file with query sequences labeled with sequences indices
Both files should follow the format below:
```
>>> 1
ATGCN...
>>> 2
TCGGA...
```
Fasta files can be downloaded from various sources such as [GenBank](https://www.ncbi.nlm.nih.gov/genbank/) or projects such as [Genome in a Bottle](https://www.nist.gov/programs-projects/genome-bottle). 







