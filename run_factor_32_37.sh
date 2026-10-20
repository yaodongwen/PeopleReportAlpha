#!/bin/bash
set -u

BASE_XML="ai_v2.xml"
XML_DIR="xml_policy_type"
LOG_DIR="logs_policy_type"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: ${BASE_XML} not found"
    exit 1
fi

FACTORS=(28 32 33 34 35 36 37)

echo "============================================================"
echo "PeopleDaily Policy-Type Factor Test"
echo "============================================================"
echo "28 = current benchmark"
echo "32 = supportive surprise + Factor28 acceleration veto"
echo "33 = restrictive surprise + Factor28 acceleration veto"
echo "34 = investment/subsidy/tax surprise + acceleration veto"
echo "35 = support/stimulus/demand surprise + acceleration veto"
echo "36 = supportive residual vs quality + acceleration veto"
echo "37 = restrictive residual vs quality + acceleration veto"
echo ""
echo "sign=-1"
echo "extremeCut=0.3"
echo "reactionWindow=5"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-ptype"
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
        "${BASE_XML}" > "${XML_FILE}"

    START_TIME=$(date +%s)

    alphasim "${XML_FILE}" > "${LOG_FILE}" 2>&1
    STATUS=$?

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [ ${STATUS} -eq 0 ]; then
        echo "[OK] ${FACTOR_ID}"
        echo "elapsed: ${ELAPSED}s"
        if [ -e "pnl/${FACTOR_ID}" ]; then
            echo ""
            simsummary "pnl/${FACTOR_ID}"
        else
            echo "WARNING: pnl/${FACTOR_ID} not found"
        fi
    else
        echo "[FAILED] ${FACTOR_ID}"
        echo "exit code: ${STATUS}"
        echo "log: ${LOG_FILE}"
        tail -50 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "ALL POLICY-TYPE TESTS FINISHED"
echo "============================================================"
