# Influence-aware Task Assignment in Spatial Crowdsourcing
This is the implementation for the ITA model architecture described in the paper: "Influence-aware Task Assignment in Spatial Crowdsourcing".

ITA studies worker-task influence on task assignment, which considers the worker-task influence based on the social network and historical task-performing patterns. Based on the worker-task influence, several influence-aware task assignment  algorithms are proposed to maximize the assigned tasks and worker-task influence. The experiments conducted on two real datasets demonstrate the effectiveness of the proposed solutions.

# Requirements

* Python 3.6
* Numpy (1.91.1)
* Scikit-learn (0.19.1)

# Dataset

* SMAP and MSL:

```
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

* SMD:

```
https://github.com/NetManAIOps/OmniAnomaly
```

* SWaT:

```
http://itrust.sutd.edu.sg/research/dataset
```

# Usage 

* Preprocess the data

```
python data_preprocess.py
```

* Run the code

```
python main.py <dataset>
```

where `<dataset>` is one of `SMAP`, `MSL`, `SMD`, `SWAT`.
