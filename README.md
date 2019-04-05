
# SENG 474 Final Project

Spring 2019 

Group 10 - 

Bing Gao V00890037

Jiaqi Huang V00862966

Zijie Li  V00863629

------

## Install required packages
```
  pip3 install -r requirements.txt
```
## Command Line
``` 
python3 main.py 
```

options:
- __frac__: range:(0,1]. The fraction of dataset to be used for training and testing. Default: 1.
- __test_size__: range:(0,1]. The proporttion of whole dataset to be used as test data. Default: 0.1.
- __data_path__: The path of dataset. Default: ./data
### Example:
```
--frac 0.1
--test_size 0.2
--data_path ./my_dataset
```

## Dataset
__station_status.csv__

Data about the weather and fullness for given station and time.

Download URL: https://drive.google.com/file/d/1wGfbixPh6riggJTVD-T0GWEOKFQ0lC3J/view?usp=sharing

__trip_difference.csv__

Data about the weather and the difference of the number of  incoming and outgoing bikes for given station and time.

Download URL: https://drive.google.com/file/d/1_kqr0r22UG-tcCh98YqzgcKK2WcQBbyX/view?usp=sharing