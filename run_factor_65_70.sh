#!/bin/bash
set -u

BASE_XML="ai.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v13.py"

XML_DIR="xml_factor65_70"
LOG_DIR="logs_factor65_70"
mkdir -p "${XML_DIR}" "${LOG_DIR}"

FACTORS=(54 61 65 66 67 68 69 70)

echo "============================================================"
echo "PeopleDaily Continuous Semantic + Combination Architecture Test"
echo "============================================================"
echo "54 = frozen benchmark"
echo "61 = reversed concentration standalone (known useful)"
echo "65 = continuous semantic novelty"
echo "66 = reversed continuous semantic novelty"
echo "67 = F54 long-side gate by F61 disagreement"
echo "68 = F54 long-side gate by F66 disagreement"
echo "69 = ONE-factor stock-sleeve combo: F54 75% + F61 25%"
echo "70 = ONE-factor stock-sleeve combo: F54 75% + F66 25%"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-arch"
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
        echo "log: ${LOG_FILE}"
        tail -160 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "CORRELATIONS TO F54"
echo "============================================================"

for F in 61 65 66 67 68 69 70
do
    if [ -e "pnl/factor${F}-arch" ]; then
        printf "corr(F54,F%s) = " "${F}"
        bcorr pnl/factor54-arch pnl/factor${F}-arch 2 2 2>/dev/null
    fi
done

echo ""
echo "============================================================"
echo "AUX CORRELATIONS"
echo "============================================================"

if [ -e "pnl/factor61-arch" ] && [ -e "pnl/factor66-arch" ]; then
    printf "corr(F61,F66) = "
    bcorr pnl/factor61-arch pnl/factor66-arch 2 2 2>/dev/null
fi

echo ""
echo "============================================================"
echo "TEST FINISHED"
echo "============================================================"
