#!/bin/bash
set -u

BASE_XML="ai_v2.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v9.py"

XML_DIR="xml_time_structure_51_54"
LOG_DIR="logs_time_structure_51_54"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: ${BASE_XML} not found"
    exit 1
fi

if [ ! -f "${ALPHA_MODULE}" ]; then
    echo "ERROR: ${ALPHA_MODULE} not found"
    exit 1
fi

TIME_NPZ="data/factor/people_daily_time_structure_factor.npz"
if [ ! -f "${TIME_NPZ}" ]; then
    echo "ERROR: ${TIME_NPZ} not found"
    echo "Run first:"
    echo "  python 06_build_time_structure_factors.py"
    exit 1
fi

FACTORS=(50 51 52 53 54)

echo "============================================================"
echo "PeopleDaily Time-Structure Factor Test"
echo "============================================================"
echo "50 = frozen benchmark: F28/F36 post-veto rank ensemble"
echo "51 = policy persistence surprise rank + accel veto"
echo "52 = policy burst(5d mean - 20d mean) surprise rank + accel veto"
echo "53 = 0.75*Rank(F50) + 0.25*Rank(F51), preserving veto zeros"
echo "54 = 0.75*Rank(F50) + 0.25*Rank(F52), preserving veto zeros"
echo ""
echo "sign=-1"
echo "extremeCut=0.3"
echo "reactionWindow=5"
echo "vetoThreshold=-0.6"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-time"
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
        tail -120 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "CORRELATION TO F50"
echo "============================================================"

if [ -e "pnl/factor50-time" ]; then
    for F in 51 52 53 54
    do
        if [ -e "pnl/factor${F}-time" ]; then
            printf "corr(F50,F%s) = " "${F}"
            bcorr \
                "pnl/factor50-time" \
                "pnl/factor${F}-time" \
                2 2 2>/dev/null
        fi
    done
fi

echo ""
echo "============================================================"
echo "ALL TIME-STRUCTURE TESTS FINISHED"
echo "============================================================"
