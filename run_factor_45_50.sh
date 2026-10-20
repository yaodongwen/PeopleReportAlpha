#!/bin/bash
set -u

BASE_XML="ai_v2.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v8.py"

XML_DIR="xml_synthesis_45_50"
LOG_DIR="logs_synthesis_45_50"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: ${BASE_XML} not found"
    exit 1
fi

if [ ! -f "${ALPHA_MODULE}" ]; then
    echo "ERROR: ${ALPHA_MODULE} not found"
    exit 1
fi

FACTORS=(28 45 46 47 48 49 50)

echo "============================================================"
echo "PeopleDaily Single-Factor Synthesis Test"
echo "============================================================"
echo "28 = benchmark"
echo "45 = F28 + 0.15*Orth(F36|F28), rank, then accel veto"
echo "46 = F28 + 0.25*Orth(F36|F28), rank, then accel veto"
echo "47 = F28 + 0.35*Orth(F36|F28), rank, then accel veto"
echo "48 = protect |F28|>=0.6; weak F28 += 0.25*F36"
echo "49 = same-direction F36 confirmation only"
echo "50 = post-veto dual-rank fusion 75/25"
echo ""
echo "sign=-1"
echo "extremeCut=0.3"
echo "reactionWindow=5"
echo "vetoThreshold=-0.6"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-synth"
    XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"
    LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"

    echo ""
    echo "============================================================"
    echo "Running ${FACTOR_ID}"
    echo "============================================================"

    sed \
        -e 's#module="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha[^"]*\.py"#module="'"${ALPHA_MODULE}"'"#g' \
        -e 's/id="factor[0-9A-Za-z+_.-]*"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
        -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
        -e 's/extremeCut="[+-]*[0-9.]*"/extremeCut="0.3"/g' \
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
        tail -100 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "ALL SYNTHESIS TESTS FINISHED"
echo "============================================================"
