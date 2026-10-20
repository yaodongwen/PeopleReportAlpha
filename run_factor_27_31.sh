#!/bin/bash

set -u


BASE_XML="ai_v2.xml"

XML_DIR="xml_long_veto"
LOG_DIR="logs_long_veto"


mkdir -p "${XML_DIR}"
mkdir -p "${LOG_DIR}"


if [ ! -f "${BASE_XML}" ]; then

    echo "ERROR: ${BASE_XML} not found"

    exit 1
fi


FACTORS=(
    22
    27
    28
    29
    30
    31
)


echo "============================================================"
echo "PeopleDaily Factor22 Long-Side Veto Test"
echo "============================================================"
echo "Factor22 = benchmark"
echo "Factor27 = RET5 weak veto"
echo "Factor28 = ACCEL5 weak veto"
echo "Factor29 = INTRADAY5 weak veto"
echo "Factor30 = OVERNIGHT5 weak veto"
echo "Factor31 = >=2 weak states veto"
echo ""
echo "sign=-1"
echo "extremeCut=0.3"
echo "reactionWindow=5"
echo "============================================================"


for FACTOR_TYPE in "${FACTORS[@]}"
do

    FACTOR_ID="factor${FACTOR_TYPE}-longveto"

    XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"

    LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"


    echo ""
    echo "============================================================"
    echo "Running ${FACTOR_ID}"
    echo "============================================================"


    sed \
        -e 's/id="factor[0-9A-Za-z+_.-]*"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/id="test\.valid"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
        -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
        -e 's/extremeCut="[0-9.]*"/extremeCut="0.3"/g' \
        -e 's/reactionWindow="[0-9]*"/reactionWindow="5"/g' \
        -e 's/debug="[0-9]*"/debug="0"/g' \
        "${BASE_XML}" \
        > "${XML_FILE}"


    START_TIME=$(date +%s)


    alphasim "${XML_FILE}" \
        > "${LOG_FILE}" \
        2>&1


    STATUS=$?


    END_TIME=$(date +%s)

    ELAPSED=$((END_TIME - START_TIME))


    if [ ${STATUS} -eq 0 ]; then

        echo "[OK] ${FACTOR_ID}"
        echo "elapsed: ${ELAPSED}s"

        if [ -e "pnl/${FACTOR_ID}" ]; then

            echo ""

            simsummary \
                "pnl/${FACTOR_ID}"

        else

            echo "WARNING: pnl/${FACTOR_ID} not found"

        fi

    else

        echo "[FAILED] ${FACTOR_ID}"
        echo "exit code: ${STATUS}"
        echo "log: ${LOG_FILE}"

        tail -50 \
            "${LOG_FILE}"

    fi

done


echo ""
echo "============================================================"
echo "ALL LONG-SIDE VETO TESTS FINISHED"
echo "============================================================"