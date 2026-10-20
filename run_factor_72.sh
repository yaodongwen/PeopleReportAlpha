#!/bin/bash
set -u

BASE_XML="ai.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v15.py"

NEW_NPZ="data/factor/people_daily_direction_consistency_factor.npz"

XML_DIR="xml_factor72"
LOG_DIR="logs_factor72"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${NEW_NPZ}" ]; then
    echo "ERROR: ${NEW_NPZ} not found"
    echo "Run first:"
    echo "  cd ai_policy_label"
    echo "  python 09_build_direction_consistency_factor.py"
    echo "  cd .."
    exit 1
fi

FACTORS=(54 72)

echo "============================================================"
echo "PeopleDaily Direction Consistency Test"
echo "============================================================"
echo "54 = frozen benchmark"
echo "72 = direction consistency standalone"
echo ""
echo "Hard requirement for candidate:"
echo "  max absolute pool correlation < 0.20"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-consistency"
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
echo "CORRELATION TO F54"
echo "============================================================"

if [ -e "pnl/factor54-consistency" ] && [ -e "pnl/factor72-consistency" ]; then
    printf "corr(F54,F72) = "
    bcorr \
        pnl/factor54-consistency \
        pnl/factor72-consistency \
        2 2 2>/dev/null
fi

echo ""
echo "============================================================"
echo "FACTOR72 RISK CORRELATIONS"
echo "============================================================"

bcorr \
    pnl/factor72-consistency \
    /mnt/nas-intern/homes/dwyao/alpha_pnl_risk_sim3/* \
    2 2

echo ""
echo "============================================================"
echo "FACTOR72 POOL EXTREMES"
echo "============================================================"

bcorr pnl/factor72-consistency /data/share/poolpnl/* 2 2 2>/dev/null \
| sort -u -g -k 1 \
| awk '
{ lines[NR]=$0 }
END {
    print "--- min ---"
    for(i=1;i<=10 && i<=NR;i++) print lines[i]
    print "--- max ---"
    start=NR-9
    if(start<1) start=1
    for(i=start;i<=NR;i++) print lines[i]
}'

echo ""
echo "============================================================"
echo "FACTOR72 CHECK FINISHED"
echo "============================================================"
