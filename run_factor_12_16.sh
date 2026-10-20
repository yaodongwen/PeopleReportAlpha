#!/bin/bash

set -u

BASE_XML="ai.xml"

XML_DIR="xml_runs"
LOG_DIR="logs"

mkdir -p "${XML_DIR}"
mkdir -p "${LOG_DIR}"


if [ ! -f "${BASE_XML}" ]; then

    echo "ERROR: missing ${BASE_XML}"
    exit 1

fi


echo "============================================================"
echo "PeopleDaily AI Factor 12-16"
echo "sign = -1"
echo "============================================================"


for FACTOR_TYPE in $(seq 12 16)
do

    FACTOR_ID="factor${FACTOR_TYPE}-1"

    XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"

    LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"

    echo ""
    echo "============================================================"
    echo "Running ${FACTOR_ID}"
    echo "factorType=${FACTOR_TYPE}"
    echo "sign=-1"
    echo "============================================================"

    sed \
        -e 's/id="factor[0-9][0-9]*-[+]*1"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/id="test\.valid"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
        -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
        -e 's/debug="[0-9]*"/debug="0"/g' \
        "${BASE_XML}" \
        > "${XML_FILE}"

    START_TIME=$(date +%s)

    alphasim "${XML_FILE}" \
        > "${LOG_FILE}" \
        2>&1

    STATUS=$?

    END_TIME=$(date +%s)

    ELAPSED=$(
        (
            END_TIME
            -
            START_TIME
        )
    )

    if [ ${STATUS} -eq 0 ]; then

        echo "[OK] ${FACTOR_ID}"
        echo "elapsed: ${ELAPSED}s"

        if [ -e "pnl/${FACTOR_ID}" ]; then

            echo ""
            simsummary \
                "pnl/${FACTOR_ID}"

        else

            echo \
                "WARNING: pnl/${FACTOR_ID} not found"

        fi

    else

        echo \
            "[FAILED] ${FACTOR_ID}"

        echo \
            "log: ${LOG_FILE}"

        tail -40 \
            "${LOG_FILE}"

    fi

done


echo ""
echo "============================================================"
echo "All factor12-16 finished"
echo "============================================================"