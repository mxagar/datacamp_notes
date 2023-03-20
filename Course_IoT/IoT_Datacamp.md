# Datacamp Course: Analyzing IoT Data in Python

These are my personal notes of the Datacamp course [Analyzing IoT Data in Python](https://app.datacamp.com/learn/courses/analyzing-iot-data-in-python).

Note that:

- Images are in [`pics`](pics).
- The code/exercises are in [`lab`](lab).
- The data is in [`data`](data).

Mikel Sagardia, 2023.  
No guarantees.

## Table of Contents

- [Datacamp Course: Analyzing IoT Data in Python](#datacamp-course-analyzing-iot-data-in-python)
  - [Table of Contents](#table-of-contents)
  - [0. Setup](#0-setup)
  - [1. Introduction to IOT Data: Accessing IoT Data](#1-introduction-to-iot-data-accessing-iot-data)
    - [1.1 Data Acquisition](#11-data-acquisition)
    - [1.2 Understanding the Data](#12-understanding-the-data)
    - [1.3 Introduction to Data Streams](#13-introduction-to-data-streams)

## 0. Setup

You should create an environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the necessary libraries/packages. A brief recipe for that:

```bash
# Create and activate e
conda create --name ds pip python=3.7
conda activate ds

# Install pip dependencies
pip install -r requirements.txt

# Track any changes and versions you have
pip list --format=freeze > requirements.txt
```

## 1. Introduction to IOT Data: Accessing IoT Data

IoT = Internet of Things: a network of connected devices sharing data from the environment; devices that collect data everywhere.

Common formats:

- JSON
- Plain text
- XML
- Binary
- Closed protocols

Data acquisition:

- Often IoT data is collected in data streams;
- Collected from devices;
- API endpoints.

### 1.1 Data Acquisition

Typical data acquisition with `requests` and `pandas`:

```python
import requests
import pandas as pd

# Option 1: Requests
url = "https://demo.datacamp.com/api/temp?count=3"
r = requests.get(url)
# Extract JSON
r.json()

# Convert JSON to pandas dataframe
df = pd.DataFrame(r.json()).head()

# Option 2: Handle download + conversion with pandas
url = "https://demo.datacamp.com/api/temp?count=3"
df_env = pd.read_json(url)
df_env.head()

# Pandas often takes care of data types, e.g., timestamps
print(df_env.dtypes)
```

To store the data:

```python
# JSON
df_env.to_json("data.json", orient="records")
# CSV
df_temp.to_csv("temperature.csv", index=False)
```

### 1.2 Understanding the Data

```python
import requests
import pandas as pd

DATA_PATH = "../data/"
filename = "environ_MS83200MS_nowind_3m-10min.json"

df = pd.read_json(DATA_PATH+filename)
df.head()

df.info()
df.describe()
```

### 1.3 Introduction to Data Streams

Installation:

```bash
python -m pip install paho-mqtt
```

[MQTT Beginners Guide](https://medium.com/python-point/mqtt-basics-with-python-examples-7c758e605d4)