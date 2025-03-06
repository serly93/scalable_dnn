## A Scalable DNN Training Framework for Traffic Forecasting in Mobile Networks

In this repository we release the code of our paper:
- **A Scalable DNN Training Framework for Traffic Forecasting in Mobile Networks** in IEEE ICMLCN 2025 - IEEE  International Conference on Machine Learning for Communication and Networking, Barcelona, Spain.

*Serly Moghadas Gholian, Claudio Fiandrino and Joerg Widmer*

If you use this code in your work, please cite our paper as follows: 
> Moghadas Gholian, S., Fiandrino, C. and Widmer, J., "A Scalable DNN Training Framework for Traffic Forecasting in Mobile Networks" In IEEE International Conference on Machine Learning for Communication and Networking, 2025, pp. 1-7

## Abstract

The exponential growth of mobile data traffic demands efficient and scalable forecasting methods to optimize network performance. Traditional approaches, like training individual models for each Base Station (BS) are computationally prohibitive for large-scale production deployments. In this paper, we propose a scalable Deep Neural Networks (DNN) training framework for mobile network traffic forecasting that reduces input redundancy and computational overhead. We minimize the number of input probes (traffic monitors at Base Stations (BSs)) by grouping BS s with temporal similarity using K-means clustering with Dynamic Time Warping (DTW) as the distance metric. Within each cluster, we train a DNN model, selecting a subset of BSs as inputs to predict future traffic demand for all BSs in that cluster. To further optimize input selection, we leverage the well-known EXplainable Artificial Intelligence (XAI) technique, LayeR-wise backPropagation (LRP) to identify the most influential BS s within each cluster. This makes it possible to reduce the number of required probes while maintaining high prediction accuracy. To validate our newly proposed framework, we conduct experiments on two real-world mobile traffic datasets. Specifically, our approach achieves competitive accuracy while reducing the total number of input probes by approximately 81% compared to state-of-the-art predictors.

## Dependencies 
- Python 3.8
- [Anaconda](https://www.anaconda.com/products/distribution). 
- jupyterlab or any IDE that supports `.ipynb` files.

## Cloning the environment
The required libararies and their compatable versions can be extracted from xai.yml file or you can directly create a conda environment which will download and install all the required packages by running `conda env create -f xai.yml` in the command line.

# Datasets
We use two datasets:\
**Milan dataset** and **EU Metropolitan Area (EUMA) dataset:**
*Unfortunately we cannot make EUMA dataset public, nevertheless, the scripts work for both cities.* 
- Each dataset provides the temporally aggregated internet activity of each cell withing it's region.
-  Milan dataset is structured as 100x100 grid area where different CDR data is captured from various base stations in the region and distributed using Voronoi-tessellation technique among all the cells.
- We only extract internet activity from these cells which is proxy of load used in each cell. The load in each cell is captured during 1 Nov 2013- 1 Jan 2014 further this load is temporally aggregated every 10 minutes. More information regarding this dataset is available [here](https://doi.org/10.1038/sdata.2015.55). 
- The EUMA dataset is structured as 48x120 grid area and contains more recent data captured in 2019 for 15 days. The load is direclty measured and is temporally aggregated every minute in each cell.
- Milan dataset is publicly available and can be accessed and downloaded from [here](https://doi.org/10.7910/DVN/EGZHFV).
-  After downloading the [dataset](https://doi.org/10.7910/DVN/EGZHFV), use the script `extract_bs.py` , to extract the internet activity from the rest of data and seperate them for each cell. 


For any questions or issues, feel free to reach out:
📧 serly.moghadas@imdea.org