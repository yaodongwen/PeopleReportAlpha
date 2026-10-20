#!/bin/bash
set -u

BASE_XML="ai.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v14.py"

XML_DIR="xml_factor69_71"
LOG_DIR="logs_factor69_71"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

FACTORS=(54 61 69 71)

echo "============================================================"
echo "PeopleDaily Stock-Sleeve Robustness Test"
echo "============================================================"
echo "54 = frozen benchmark"
echo "61 = reversed concentration standalone"
echo "69 = 75/25 stock-sleeve: F54 + F61"
echo "71 = 50/50 stock-sleeve: F54 + F61"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-robust"
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
echo "PAIRWISE CORRELATIONS"
echo "============================================================"

for PAIR in "54 61" "54 69" "54 71" "61 69" "61 71" "69 71"
do
    set -- ${PAIR}
    A=$1
    B=$2

    if [ -e "pnl/factor${A}-robust" ] && [ -e "pnl/factor${B}-robust" ]; then
        printf "corr(F%s,F%s) = " "${A}" "${B}"
        bcorr \
            "pnl/factor${A}-robust" \
            "pnl/factor${B}-robust" \
            2 2 2>/dev/null
    fi
done

echo ""
echo "============================================================"
echo "FACTOR69 RISK CORRELATIONS"
echo "============================================================"

bcorr \
    pnl/factor69-robust \
    /mnt/nas-intern/homes/dwyao/alpha_pnl_risk_sim3/* \
    2 2

echo ""
echo "============================================================"
echo "FACTOR69 POOL EXTREMES"
echo "============================================================"

bcorr pnl/factor69-robust /data/share/poolpnl/* 2 2 2>/dev/null \
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
echo "FACTOR71 RISK CORRELATIONS"
echo "============================================================"

bcorr \
    pnl/factor71-robust \
    /mnt/nas-intern/homes/dwyao/alpha_pnl_risk_sim3/* \
    2 2

echo ""
echo "============================================================"
echo "FACTOR71 POOL EXTREMES"
echo "============================================================"

bcorr pnl/factor71-robust /data/share/poolpnl/* 2 2 2>/dev/null \
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
echo "ROBUSTNESS TEST FINISHED"
echo "============================================================"
