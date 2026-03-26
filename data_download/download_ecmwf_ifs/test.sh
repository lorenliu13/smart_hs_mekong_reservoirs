#!/bin/bash

# 1. Define your fixed parameters (Area, Grid, Levels, etc.)
AREA="73.5/-27/33/45"
GRID="0.5/0.5"
PARAMS="130/131/132"
LEVELIST="127/128/129/130"
STEPS="0/to/90/by/1"

# 2. Define your time range
YEAR="2024"
MONTH="01"
TIMES="0000 1200"

# 3. Start the Loops
for my_day in $(seq -w 1 31); do
    DATE="${YEAR}${MONTH}${my_day}"

    for my_time in ${TIMES}; do
        # Define the filename for this specific chunk
        REQUEST_FILE="req_${DATE}_${my_time}.mars"
        TARGET_LIST="list_${DATE}_${my_time}.txt"

        echo "Creating LIST (cost) request for $DATE at $my_time..."

        # 4. Create the MARS LIST request file on the fly
        cat << EOF > $REQUEST_FILE
LIST,
    OUTPUT   = COST,
    CLASS    = OD,
    TYPE     = FC,
    STREAM   = OPER,
    EXPVER   = 0001,
    LEVTYPE  = ML,
    GRID     = ${GRID},
    AREA     = ${AREA},
    PARAM    = ${PARAMS},
    LEVELIST = ${LEVELIST},
    STEP     = ${STEPS},
    DATE     = ${DATE},
    TIME     = ${my_time},
    TARGET   = "${TARGET_LIST}"
EOF

        # 5. Execute the MARS LIST command
        mars $REQUEST_FILE

        # 6. Cleanup the temporary request file and show cost summary
        if [ $? -eq 0 ]; then
            rm $REQUEST_FILE
            echo "Cost estimate for $DATE $my_time written to $TARGET_LIST"
            cat "$TARGET_LIST"
        else
            echo "Error running LIST for $DATE $my_time"
        fi
    done
done
