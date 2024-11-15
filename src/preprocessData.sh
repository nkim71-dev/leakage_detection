#!/bin/bash

for i in "202304" "202305" "202306" "202307" "202308"; do
    python ./src/dataCleaning.py --month ${i}
done
python ./src/dataMerging.py
python ./src/dataLabeling.py
python ./src/columnOrganizing.py
python ./src/dataProcessing.py