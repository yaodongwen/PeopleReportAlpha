#!/bin/bash

set -u

BASE_XML="ai_v2.xml"
XML_DIR="xml_price_runs"
LOG_DIR="logs_price"

mkdir -p "${XML_DIR}"
mkdir -p "${LOG_DIR}"

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: ${BASE_XML} not found"
    exit 1
fi


# ============================================================
# Format:
#
# label|priceMode|window|weight|boost|penalty
# ============================================================

TESTS=(

    "base|0|60|0.00|0.00|0.00"

    "residual|1|60|0.00|0.00|0.00"

    "blend20w15|2|20|0.15|0.00|0.00"

    "blend20w30|2|20|0.30|0.00|0.00"

    "blend60w15|2|60|0.15|0.00|0.00"

    "blend60w30|2|60|0.30|0.00|0.00"

    "blend120w20|2|120|0.20|0.00|0.00"

    "condition60mild|3|60|0.00|0.50|0.50"

    "condition60strong|3|60|0.00|1.00|0.75"
)


echo "============================================================"
echo "PeopleDaily + Industry Price Test"
echo "factorType=22"
echo "sign=-1"
echo "extremeCut=0.3"
echo "============================================================"


for TEST in "${TESTS[@]}"
do

    IFS='|' read \
        LABEL \
        MODE \
        WINDOW \
        WEIGHT \
        BOOST \
        PENALTY \
        <<< "${TEST}"


    FACTOR_ID="pd22-${LABEL}"

    XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"
    LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"


    echo ""
    echo "============================================================"
    echo "Running ${FACTOR_ID}"
    echo "priceMode=${MODE}"
    echo "window=${WINDOW}"
    echo "weight=${WEIGHT}"
    echo "boost=${BOOST}"
    echo "penalty=${PENALTY}"
    echo "============================================================"


    sed \
        -e 's/id="factor[0-9A-Za-z+_.-]*"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/id="test\.valid"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/factorType="[0-9][0-9]*"/factorType="22"/g' \
        -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
        -e 's/extremeCut="[0-9.]*"/extremeCut="0.3"/g' \
        -e 's/priceMode="[0-9]*"/priceMode="'"${MODE}"'"/g' \
        -e 's/momentumWindow="[0-9]*"/momentumWindow="'"${WINDOW}"'"/g' \
        -e 's/priceWeight="[0-9.]*"/priceWeight="'"${WEIGHT}"'"/g' \
        -e 's/confirmBoost="[0-9.]*"/confirmBoost="'"${BOOST}"'"/g' \
        -e 's/conflictPenalty="[0-9.]*"/conflictPenalty="'"${PENALTY}"'"/g' \
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
        echo "log: ${LOG_FILE}"

        tail -40 \
            "${LOG_FILE}"

    fi

done


echo ""
echo "============================================================"
echo "ALL PRICE TESTS FINISHED"
echo "============================================================"