import chardet

with open("Using_0407_1.csv", "rb") as f:
    rawdata = f.read(10000)
    result = chardet.detect(rawdata)
    print(result)
