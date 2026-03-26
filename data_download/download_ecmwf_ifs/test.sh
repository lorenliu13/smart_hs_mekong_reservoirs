#!/bin/bash

cat << EOF > req_list.mars
LIST,
    CLASS    = OD,
    TYPE     = FC,
    STREAM   = OPER,
    EXPVER   = 0001,
    LEVTYPE  = ML,
    LEVELIST = 127/128/129/130/131/132/133/134/135/136/137,
    PARAM    = 130/131/132,
    DATE     = 20170401,
    TIME     = 0000,
    STEP     = 0/to/90/by/1,
    OUTPUT   = cost,
    TARGET   = "list2.txt"
EOF

mars req_list.mars
rm req_list.mars

echo "--- Cost output ---"
cat list2.txt
