# adu-features

Ensure you have python3.6 and pip installed

To run on your localhost follow these instructions

1. Run `git clone https://github.com/defineapoorv/adu-features.git`
2. Run `cd adu-features`
3. Run `pip install -r requirements.txt`
4. Run `python application.py`

This app currently runs only for Alameda County

On your browser run this
`http://127.0.0.1:5000/get_predictions/<pid>`

Where `pid` is Parcel id of property
Here are some examples

http://127.0.0.1:5000/get_predictions/74-437-25

http://127.0.0.1:5000/get_predictions/73-403-27

http://127.0.0.1:5000/get_predictions/74-1290-138


#Here is a Sample Output

![Sample Output](https://raw.githubusercontent.com/defineapoorv/adu-features/master/sample/test-output.png)

`pool` tells probability of property having a swimming pool

`solar` tells probability of property having a solar panel

`adu` tells probability of property having a accessory dwelling unit

