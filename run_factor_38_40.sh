#!/bin/bash
set -u

BASE_XML="ai_v2.xml"
XML_DIR="xml_combo_28_36"
LOG_DIR="logs_combo_28_36"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: ${BASE_XML} not found"
    exit 1
fi

FACTORS=(28 36 38 39 40)

echo "============================================================"
echo "PeopleDaily Factor28 + Factor36 Combination Test"
echo "============================================================"
echo "28 = benchmark: Factor22 + acceleration long veto"
echo "36 = supportive residual + acceleration long veto"
echo "38 = 0.85*F28 + 0.15*F36, then CS rerank"
echo "39 = 0.75*F28 + 0.25*F36, then CS rerank"
echo "40 = 0.65*F28 + 0.35*F36, then CS rerank"
echo ""
echo "All source signals use:"
echo "  sign=-1"
echo "  extremeCut=0.3"
echo "  vetoThreshold=-0.6"
echo "  reactionWindow=5"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-combo"
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
        -e 's/vetoThreshold="[+-]*[0-9.]*"/vetoThreshold="-0.6"/g' \
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
        tail -80 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "ALL FACTOR28 + FACTOR36 COMBINATION TESTS FINISHED"
echo "============================================================"
