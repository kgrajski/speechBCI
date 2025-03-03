# SpeechBCI

The purpose of this repo is to explore an open-source data set comprised of electrode array recordings of nueral activity during speech generation.

* Willett, et al. (2023). A high-performance neural speech prosthesis. Nature. 620:1031-1036.
* https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq

## Description

The following set of slides describes this project.
[Speech Decoding Pilot Study (Kamil Grajski 28Feb2025).pdf](https://github.com/user-attachments/files/19057861/Speech.Decoding.Pilot.Study.Kamil.Grajski.28Feb2025.pdf)

This animation is from a single trial during which the subject "spoke" a sentence.
Each point in the grid corresponds to an electrode.  The data is shown as a heat map.
[Example Animation](figs/competitionData/train/t12.2022.05.05_1_0_implot.html)

This image is from the same animation as above, but shown as a time series.
[Example Time Series](figs/competitionData/train/t12.2022.05.05_1_0_Ventral_tsplot.html)

Note: If you get the message that git cannot display such large html then download the figs folder and display locally.

## Getting Started

Assuming that one has gained access to the dataset, there are two stages to using this code.
* The ETL stage is implemented in the **etl.py**d script.
     This dices and slices and rearranges the raw data based on the Willett paper and data set README.
     This is a necessary step to make sure that processing and display can be mapped backed to physical location of the electrode.
* The VQ-VAE stage is implemented in the **dev_vqvae.py** script.
     This script manages the training, testing, and validation loop for the ETL data.
     The script leverages GPU if available and puts results to TensorBoard.

### Dependencies

* No special requirements beyond the imports listed in the scripts.
* This repo was developed and executed on AWS via VSS Code.

### Installing

* No special requirements beyond the imports listed in the scripts.
* This repo was developed and executed on AWS via VSS Code.

### Executing program

* See Getting Started above.

## Help

Send an email to: kgrajski@nurosci.com

## Authors

Kamil A. Grajski (kgrajski@nurosci.com)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

It is fantastic that Willett lab made available a dataset!

