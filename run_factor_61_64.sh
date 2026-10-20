#!/bin/bash
set -u

BASE_XML="ai.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v12.py"

XML_DIR="xml_factor61_64"
LOG_DIR="logs_factor61_64"
mkdir -p "${XML_DIR}" "${LOG_DIR}"

SEM_NPZ="data/factor/people_daily_semantic_novelty_factor.npz"

if [ ! -f "${SEM_NPZ}" ]; then
    echo "ERROR: ${SEM_NPZ} not found"
    echo "Run first:"
    echo "  cd ai_policy_label"
    echo "  python 08_build_semantic_novelty_factor.py"
    echo "  cd .."
    exit 1
fi

FACTORS=(54 61 62 63 64)

echo "============================================================"
echo "PeopleDaily Reverse Concentration + Semantic Novelty Test"
echo "============================================================"
echo "54 = frozen benchmark"
echo "61 = reversed concentration standalone"
echo "62 = 0.75*Rank(F54) + 0.25*Rank(F61)"
echo "63 = semantic first-appearance standalone"
echo "64 = 0.75*Rank(F54) + 0.25*Rank(F63)"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-next"
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

    alphasim "${XML_FILE}" > "${LOG_FILE}" 2>&1
    STATUS=$?

    if [ ${STATUS} -eq 0 ]; then
        echo "[OK] ${FACTOR_ID}"
        simsummary "pnl/${FACTOR_ID}"
    else
        echo "[FAILED] ${FACTOR_ID}"
        tail -150 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "CORRELATIONS TO F54"
echo "============================================================"

for F in 61 62 63 64
do
    if [ -e "pnl/factor${F}-next" ]; then
        printf "corr(F54,F%s) = " "${F}"
        bcorr pnl/factor54-next pnl/factor${F}-next 2 2 2>/dev/null
    fi
done

if [ -e "pnl/factor61-next" ] && [ -e "pnl/factor63-next" ]; then
    printf "corr(F61,F63) = "
    bcorr pnl/factor61-next pnl/factor63-next 2 2 2>/dev/null
fi

echo "============================================================"
echo "TEST FINISHED"
echo "============================================================"
